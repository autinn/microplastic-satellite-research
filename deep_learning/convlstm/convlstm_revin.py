import sys
# print(sys.executable) # Commented out for cleaner output

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import json
import wandb
from utils.repro_eval import set_seed, evaluate_arrays

# Set device and seed
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')
set_seed(42)
print(f"Using device: {device}")

# --- Configuration ---
# Set the path to your NetCDF files
# Make sure your 'data' folder is in the same directory as this script
NC_FILES_PATH = Path("./data")
# Using the same max files as your comparison setup
MAX_FILES_TO_PROCESS = 500 # Keep consistent with other models for comparison
SEQ_LENGTH = 4 # Sequence length for input context
TEST_RATIO = 0.2 # Test set ratio

# Japan region coordinates (broader)
JAPAN_SW_LAT, JAPAN_SW_LON = 25.35753, 118.85766
JAPAN_NE_LAT, JAPAN_NE_LON = 36.98134, 145.47117

# --- NEW: RevIN Module ---
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Reversible Instance Normalization (RevIN) module.
        Normalizes each instance (sequence) independently.
        :param num_features: the number of features (C*H*W for image data flattened per time step)
        :param eps: a small value added to the denominator for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        
        # Store mean and stdev for denormalization
        # These will be updated during the forward 'norm' pass
        self.register_buffer('mean', torch.zeros(1, 1, num_features))
        self.register_buffer('stdev', torch.ones(1, 1, num_features))

    def forward(self, x, mode: str):
        # x: (batch_size, seq_len, num_features)
        if mode == 'norm':
            self._get_statistics(x)
            x = x - self.mean
            x = x / (self.stdev + self.eps)
            if self.affine:
                x = x * self.affine_weight
                x = x + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps)
            x = x * self.stdev
            x = x + self.mean
            return x
        else:
            raise NotImplementedError(f"RevIN mode '{mode}' not supported. Use 'norm' or 'denorm'.")

    def _get_statistics(self, x):
        # x: (batch_size, seq_len, num_features)
        # Calculate mean and stdev over the sequence length (dim=1) for each batch sample and each feature
        # detach() is used to prevent gradients from flowing through these statistics calculations
        dim2reduce = 1 
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()


# --- ConvLSTM Modules (Provided by User) ---
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                each element of the list is a tuple (h, c) for hidden state and memory
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError("Stateful ConvLSTM is not implemented yet.")
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# --- Data Loading and Preprocessing Functions (Adapted from lstm_japan.py) ---

def extract_timestamps_from_filenames(nc_files_list):
    """Extract timestamps from NetCDF filenames."""
    timestamps = []
    for file in nc_files_list:
        filename = Path(file).name
        date_part = filename.split('.')[2][1:9]  # Extract sYYYYMMDD -> YYYYMMDD
        timestamp = datetime.strptime(date_part, '%Y%m%d')
        timestamps.append(timestamp)
    return timestamps

def lat_lon_to_indices(lat, lon, lats, lons):
    """Convert lat/lon coordinates to grid indices"""
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx

def process_and_normalize_image(nc_file):
    """
    Convert a netCDF file to a Japan region cropped normalized array.
    This function is similar to process_and_plot_to_array but doesn't downsample here.
    """
    ds = xr.open_dataset(nc_file)
    data = ds['mp_concentration']
    data_array_2d = data.values.squeeze() # Ensure 2D
    
    lats = None
    lons = None
    
    possible_lat_names = ['lat', 'latitude', 'y', 'lat_1', 'lat_2']
    possible_lon_names = ['lon', 'longitude', 'x', 'lon_1', 'lon_2']
    
    for lat_name in possible_lat_names:
        if lat_name in ds.variables:
            lats = ds[lat_name].values
            break
    for lon_name in possible_lon_names:
        if lon_name in ds.variables:
            lons = ds[lon_name].values
            break
    
    if lats is not None and lons is not None:
        japan_sw_lat_idx, japan_sw_lon_idx = lat_lon_to_indices(JAPAN_SW_LAT, JAPAN_SW_LON, lats, lons)
        japan_ne_lat_idx, japan_ne_lon_idx = lat_lon_to_indices(JAPAN_NE_LAT, JAPAN_NE_LON, lats, lons)
        
        japan_lat_start = min(japan_sw_lat_idx, japan_ne_lat_idx)
        japan_lat_end = max(japan_sw_lat_idx, japan_ne_lat_idx)
        japan_lon_start = min(japan_sw_lon_idx, japan_ne_lon_idx)
        japan_lon_end = max(japan_sw_lon_idx, japan_ne_lon_idx)
        
        cropped_data = data_array_2d[japan_lat_start:japan_lat_end, japan_lon_start:japan_lon_end]
    else:
        print(f"Warning: No coordinate information found in {nc_file.name}, using full dataset.")
        cropped_data = data_array_2d
    
    # Normalize the data to 0-1 range
    data_min = np.nanmin(cropped_data)
    data_max = np.nanmax(cropped_data)
    if data_max > data_min:
        normalized_data = (cropped_data - data_min) / (data_max - data_min)
    else:
        normalized_data = np.zeros_like(cropped_data)
    
    normalized_data = np.nan_to_num(normalized_data, nan=0.0)
    
    return normalized_data

def load_and_preprocess_data():
    """
    Loads all relevant NetCDF files, extracts Japan region data,
    and ensures all images are resized to a consistent shape.
    """
    nc_files_list = sorted(NC_FILES_PATH.glob("cyg.ddmi*.nc"))
    print(f"Found {len(nc_files_list)} NetCDF files.")
    
    nc_files_limited = nc_files_list[:MAX_FILES_TO_PROCESS]
    print(f"Processing {len(nc_files_limited)} files (limited from {len(nc_files_list)} total files)")
    
    processed_images = []
    
    # Process first file to get target shape
    first_image = process_and_normalize_image(nc_files_limited[0])
    # Downsample to 128x128 as per CNN-LSTM script
    target_height, target_width = 128, 128 
    # Use skimage.transform.resize (as in lstm_japan.py)
    from skimage.transform import resize 
    first_image_resized = resize(first_image, (target_height, target_width), preserve_range=True)
    processed_images.append(first_image_resized)

    print(f"Target image resolution set to: {target_height}x{target_width}")
    
    for i, nc_file in enumerate(nc_files_limited[1:]): # Start from second file
        if (i + 1) % 100 == 0:
            print(f"Processed {i+2}/{len(nc_files_limited)} files...")
        
        img = process_and_normalize_image(nc_file)
        # Ensure all images are resized to the consistent target_height, target_width
        img_resized = resize(img, (target_height, target_width), preserve_range=True)
        processed_images.append(img_resized)
            
    data = np.array(processed_images)
    print(f"Final preprocessed data shape: {data.shape}") # (num_samples, H, W)
    return data, extract_timestamps_from_filenames(nc_files_limited)

def create_convlstm_sequences(data, seq_length):
    """
    Create sequences for ConvLSTM.
    ConvLSTM expects input of shape (batch, T, C, H, W).
    Here, C=1 (single channel image).
    """
    sequences = []
    target_images = []
    
    num_samples, img_height, img_width = data.shape

    for i in range(num_samples - seq_length):
        seq = data[i : i + seq_length] # (seq_length, H, W)
        target_image = data[i + seq_length] # (H, W)
        
        # Add channel dimension: (seq_length, 1, H, W)
        seq_with_channel = seq[:, np.newaxis, :, :]
        
        sequences.append(seq_with_channel)
        target_images.append(target_image)
    
    X = np.array(sequences) # (num_sequences, seq_length, 1, H, W)
    Y = np.array(target_images) # (num_sequences, H, W)
    
    print(f"Created ConvLSTM sequences:")
    print(f"X (context) shape: {X.shape}")
    print(f"Y (target) shape: {Y.shape}")
    
    return X, Y, img_height, img_width


def temporal_train_test_split_proper(data, seq_length, test_ratio=0.2):
    """
    Properly split time series data to avoid leakage.
    
    Key insight: We need to leave a gap of seq_length between train and test
    to ensure no training targets appear in test sequences.
    """
    total_files = len(data)
    
    # Calculate split points with gap
    test_files_needed = int(total_files * test_ratio)
    gap_needed = seq_length  # Need gap equal to sequence length
    
    # Ensure we have enough data
    min_train_files = seq_length + 1  
    min_files_needed = min_train_files + gap_needed + test_files_needed + seq_length
    
    if total_files < min_files_needed:
        print(f"WARNING: Only {total_files} files available, need {min_files_needed} for proper split")
        print("Reducing test ratio to fit available data...")
        test_files_needed = max(1, (total_files - min_train_files - gap_needed - seq_length) // 2)
    
    train_end = total_files - test_files_needed - gap_needed - seq_length
    test_start = train_end + gap_needed
    
    print(f"\n=== PROPER TEMPORAL SPLIT ===")
    print(f"Total files: {total_files}")
    print(f"Training files: 0 to {train_end-1} ({train_end} files)")
    print(f"Gap (no data used): {train_end} to {test_start-1} ({gap_needed} files)")
    print(f"Test files: {test_start} to {total_files-1} ({total_files - test_start} files)")
    
    # Split the data
    train_data = data[:train_end]
    test_data = data[test_start:]
    
    return train_data, test_data, train_end, test_start


# --- ConvLSTM Model Definition ---

class ConvLSTMModel(nn.Module):
    """
    ConvLSTM model for spatiotemporal prediction.
    Replaces the CNN encoder and standard LSTM from EfficientCNNLSTMModel.
    """
    def __init__(self, seq_length, img_height, img_width, 
                 input_channels=1, hidden_channels=[64, 64], kernel_size=(3, 3), num_layers=2):
        super(ConvLSTMModel, self).__init__()
        
        self.seq_length = seq_length
        self.img_height = img_height
        self.img_width = img_width
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # NEW: RevIN layer
        # num_features for RevIN is C*H*W, as we normalize each pixel's time series
        self.revin = RevIN(num_features=self.input_channels * self.img_height * self.img_width)

        # Instantiate the ConvLSTM module
        self.convlstm = ConvLSTM(input_dim=self.input_channels,
                                 hidden_dim=self.hidden_channels,
                                 kernel_size=self.kernel_size,
                                 num_layers=self.num_layers,
                                 batch_first=True, # Input will be (B, T, C, H, W)
                                 bias=True,
                                 return_all_layers=False) # We only need the output of the last layer

        # Decoder to map ConvLSTM output (hidden state) back to image
        self.decoder_conv = nn.Conv2d(in_channels=self.hidden_channels[-1],
                                      out_channels=1,
                                      kernel_size=1, # 1x1 conv to change channels
                                      padding=0)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, 1, height, width)
        batch_size, seq_length, channels, height, width = x.shape

        # NEW: Apply RevIN normalization
        # Reshape for RevIN: (B, T, C*H*W)
        x_flattened_for_revin = x.reshape(batch_size, seq_length, channels * height * width)
        x_normalized = self.revin(x_flattened_for_revin, 'norm')
        # Reshape back to (B, T, C, H, W) for ConvLSTM
        x_normalized = x_normalized.reshape(batch_size, seq_length, channels, height, width)
        
        # Pass through ConvLSTM
        layer_output_list, last_state_list = self.convlstm(x_normalized)
        
        last_convlstm_output = last_state_list[-1][0] # Get the hidden state of the last layer
        
        # Pass through decoder convolutional layer
        output = self.decoder_conv(last_convlstm_output) # (B, 1, H, W)
        
        # Remove channel dimension for final image: (B, H, W)
        output = output.squeeze(1)
        
        # NEW: Apply RevIN denormalization
        # The output is (B, H, W), but RevIN denorms (B, 1, num_features)
        # So, we reshape the output for denormalization
        output_flattened_for_revin = output.reshape(batch_size, 1, height * width) # (B, 1, H*W)
        output_denormalized = self.revin(output_flattened_for_revin, 'denorm')
        # Reshape back to (B, H, W)
        output_denormalized = output_denormalized.reshape(batch_size, height, width)

        # Apply final activation and scaling (similar to CNN-LSTM)
        # Note: If RevIN denormalizes to original scale, tanh/sigmoid might not be needed
        # if the original data wasn't strictly 0-1 or -1-1.
        # However, since your original data is normalized 0-1, we keep this.
        output_final = torch.tanh(output_denormalized)  # Range [-1, 1]
        output_final = (output_final + 1) / 2    # Scale to [0, 1]
        
        return output_final

# --- Loss function (Copied from lstm_japan.py) ---
def improved_contrast_loss(pred, target, low_threshold=0.2, high_threshold=0.6, 
                          low_weight=2.0, high_weight=2.0, contrast_weight=1.0):
    """
    Improved loss function that balances high/low concentrations and encourages contrast.
    """
    # Basic MAE
    mae = torch.mean(torch.abs(pred - target))
    
    # Create masks for different concentration levels
    low_concentration_mask = (target < low_threshold).float()
    high_concentration_mask = (target > high_threshold).float()
    mid_concentration_mask = 1.0 - low_concentration_mask - high_concentration_mask
    
    # Weighted loss that pays attention to ALL concentration levels
    weighted_mae = torch.mean(
        torch.abs(pred - target) * (
            1.0 +  # Base weight
            low_weight * low_concentration_mask +      # Extra attention to low concentrations
            high_weight * high_concentration_mask +    # Extra attention to high concentrations
            0.5 * mid_concentration_mask               # Some attention to mid concentrations
        )
    )
    
    # Contrast preservation loss - encourages maintaining the range of values
    pred_std = torch.std(pred)
    target_std = torch.std(target)
    contrast_loss = torch.abs(pred_std - target_std)
    
    # Edge preservation loss - maintains sharp transitions
    pred_grad_x = torch.abs(pred[:, :-1, :] - pred[:, 1:, :])
    target_grad_x = torch.abs(target[:, :-1, :] - target[:, 1:, :])
    pred_grad_y = torch.abs(pred[:, :, :-1] - pred[:, :, 1:])
    target_grad_y = torch.abs(target[:, :, :-1] - target[:, :, 1:])
    
    edge_loss = torch.mean(torch.abs(pred_grad_x - target_grad_x)) + \
                torch.mean(torch.abs(pred_grad_y - target_grad_y))
    
    # Combine all losses
    total_loss = weighted_mae + contrast_weight * contrast_loss + 0.1 * edge_loss
    
    return total_loss

# --- Training and Evaluation Functions (Adapted from lstm_japan.py) ---

def train_convlstm_model(model, train_loader, val_loader, epochs=5, learning_rate=0.001):
    """Train the ConvLSTM model efficiently with progress monitoring."""
    print(f"\nStarting ConvLSTM model training for {epochs} epochs...")
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if batch_idx % 5 == 0: # Print progress every 5 batches
                print(f"  Training batch {batch_idx+1}/{len(train_loader)}")
            
            batch_x = batch_x.to(device) # (B, T, C, H, W)
            batch_y = batch_y.to(device) # (B, H, W)
            
            optimizer.zero_grad()
            outputs = model(batch_x) # (B, H, W)
            
            loss = improved_contrast_loss(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        print(f"  Starting validation...")
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = improved_contrast_loss(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"  Epoch {epoch+1} completed - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    print(f"\nTraining completed successfully!")
    
    return history

def evaluate_model(model, test_loader, timestamps, test_start_idx):
    """Evaluate the model, create visualizations, and save data for GIF."""
    print("\nEvaluating model...")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate overall metrics
    overall_mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    
    ssim_scores = []
    # Calculate individual MAE for plotting and saving
    individual_maes = np.array([mean_absolute_error(targets[i].flatten(), predictions[i].flatten()) for i in range(len(predictions))])

    for i in range(len(predictions)):
        pred_img = predictions[i]
        target_img = targets[i]
        
        pred_img_norm = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
        target_img_norm = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
        
        ssim_score = ssim(target_img_norm, pred_img_norm, data_range=1.0)
        ssim_scores.append(ssim_score)
    
    mean_ssim = np.mean(ssim_scores)
    
    print(f"Overall Test MAE: {overall_mae:.6f}")
    print(f"Overall Mean SSIM: {mean_ssim:.4f}")
    
    # --- Save data for GIF generation ---
    test_timestamps_aligned = timestamps[test_start_idx : test_start_idx + len(targets)]

    np.save('convlstm_revin_predictions.npy', predictions)
    np.save('convlstm_revin_targets.npy', targets)
    # Save per-frame absolute error maps for downstream visualization
    try:
        error_maps = np.abs(predictions - targets)
        np.save('convlstm_revin_error_maps.npy', error_maps)
    except Exception as e:
        print(f"Warning: failed to save convlstm_revin error maps: {e}")
    np.save('convlstm_revin_individual_maes.npy', individual_maes)
    
    with open('convlstm_revin_timestamps.json', 'w') as f:
        json.dump([ts.strftime("%Y-%m-%d") for ts in test_timestamps_aligned], f)

    print("\nSaved ConvLSTM predictions, targets, MAEs, and timestamps for GIF generation.")

    # --- Create the first visualization (first 4 predictions) ---
    num_samples_to_plot = 8 # Total number of samples to display
    indices_to_plot = []
    
    if len(predictions) >= num_samples_to_plot:
        indices_to_plot.extend(range(4)) # First 4
        indices_to_plot.extend(range(len(predictions) - 4, len(predictions))) # Last 4
    else: 
        indices_to_plot.extend(range(len(predictions)))
    
    indices_to_plot = sorted(list(set(indices_to_plot)))
    
    fig, axes = plt.subplots(2, len(indices_to_plot), figsize=(4 * len(indices_to_plot), 10))
    
    model_type = "ConvLSTM"
    fig.suptitle(f'{model_type} Predictions vs Actual (Japan Region)\n(Covering Japan, Korea, and Eastern China)', fontsize=16)
    
    if len(indices_to_plot) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, i in enumerate(indices_to_plot):
        target_flipped = np.flipud(targets[i])
        axes[0, col_idx].imshow(target_flipped, cmap='viridis', aspect='auto')
        
        current_timestamp = test_timestamps_aligned[i]
        
        axes[0, col_idx].set_title(f'Actual\n{current_timestamp.strftime("%Y-%m-%d")}\nMAE: {individual_maes[i]:.4f}')
        axes[0, col_idx].axis('off')
        
        pred_flipped = np.flipud(predictions[i])
        axes[1, col_idx].imshow(pred_flipped, cmap='viridis', aspect='auto')
        axes[1, col_idx].set_title(f'Predicted\nMAE: {individual_maes[i]:.4f}')
        axes[1, col_idx].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig('convlstm_revin_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return overall_mae, mean_ssim, predictions

# --- Main Execution ---

if __name__ == "__main__":
    print("=" * 60)
    print("JAPAN REGION CONVLSTM FOR MICROPLASTIC CONCENTRATION")
    print("=" * 60)
    
    # Extract data
    print("\nLoading and preprocessing data...")
    all_data_images, timestamps_full = load_and_preprocess_data()
    
    # Proper temporal split of the full image data
    train_data_images, test_data_images, train_end_idx, test_start_idx = temporal_train_test_split_proper(
        all_data_images, SEQ_LENGTH, TEST_RATIO
    )
    
    # Create ConvLSTM sequences separately for train and test
    X_train, y_train, img_height, img_width = create_convlstm_sequences(train_data_images, SEQ_LENGTH)
    X_test, y_test, _, _ = create_convlstm_sequences(test_data_images, SEQ_LENGTH)
    
    print(f"\nFinal data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Create validation split from training data (temporal split)
    val_split_idx = int(len(X_train) * 0.8)
    X_train_final = X_train[:val_split_idx]
    y_train_final = y_train[:val_split_idx]
    X_val = X_train[val_split_idx:]
    y_val = y_train[val_split_idx:]
    
    print(f"\nValidation split:")
    print(f"Final training: {X_train_final.shape}")
    print(f"Validation: {X_val.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_final)
    y_train_tensor = torch.FloatTensor(y_train_final)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    batch_size = 8 # Same batch size as CNN-LSTM for consistency
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize wandb
    wandb.init(
        project="mp-prediction-convlstm", # New project name for ConvLSTM
        config={
            "model_type": "ConvLSTM",
            "framework": "PyTorch",
            "sequence_length": SEQ_LENGTH,
            "learning_rate": 0.001,
            "batch_size": batch_size,
            "epochs": 10,
            "image_resolution": f"{img_height}x{img_width}",
            "input_channels": 1,
            "hidden_channels": [64, 64], # ConvLSTM hidden channels
            "kernel_size": (3, 3),
            "num_layers": 2, # Number of ConvLSTM layers
            "loss_function": "improved_contrast_loss",
            "geographic_coverage": "Japan_Korea_EasternChina",
            "data_files_processed": MAX_FILES_TO_PROCESS
        }
    )
    
    # Create model
    model = ConvLSTMModel(seq_length=SEQ_LENGTH, 
                          img_height=img_height, 
                          img_width=img_width,
                          input_channels=1, 
                          hidden_channels=[64, 64], # Example: 2 ConvLSTM layers with 64 hidden channels
                          kernel_size=(3, 3),
                          num_layers=2)
    
    print("\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = train_convlstm_model(model, train_loader, val_loader, epochs=10)
    
    # Evaluate model
    # Note: timestamps_full is used because test_start_idx refers to the index in the original full timestamps list.
    mae, mean_ssim, predictions = evaluate_model(model, test_loader, timestamps_full, test_start_idx) 
    metrics = evaluate_arrays(np.load('convlstm_revin_targets.npy'), predictions, tol=0.10)
    print(f"Metrics (convlstm_revin): R2={metrics['r2']:.4f} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} Acc@10%={metrics['acc_within_10pct']:.3f}")
    
    # Log final test metrics to wandb
    wandb.log({
        "test_mae": mae,
        "test_ssim": mean_ssim,
        "test_r2": metrics['r2'],
        "test_rmse": metrics['rmse'],
        "test_acc10": metrics['acc_within_10pct'],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    # Save simple training history plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('ConvLSTM Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], 'o-', label='Train')
    plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], 's-', label='Validation')
    plt.title('Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('convlstm_revin_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'model_type': 'ConvLSTM',
        'framework': 'PyTorch',
        'device': str(device),
        'sequence_length': SEQ_LENGTH,
        'image_resolution': f"{img_height}x{img_width}",
        'input_channels': 1,
        'hidden_channels': [64, 64],
        'kernel_size': (3, 3),
        'num_layers': 2,
        'training_samples': len(train_loader.dataset),
        'validation_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'test_mae': mae,
        'test_ssim': mean_ssim,
        'model_parameters': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'convlstm_hidden_channels': [64, 64],
            'convlstm_kernel_size': (3, 3),
            'convlstm_num_layers': 2
        },
        'training_config': {
            'epochs': 10,
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        },
        'geographic_region': {
            'name': 'Japan',
            'southwest': [JAPAN_SW_LAT, JAPAN_SW_LON],
            'northeast': [JAPAN_NE_LAT, JAPAN_NE_LON],
            'coverage': 'Japan, Korea, and Eastern China maritime regions'
        }
    }
    
    with open('convlstm_revin_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("JAPAN REGION CONVLSTM TRAINING COMPLETED")
    print("=" * 60)
    print(f"✓ Test MAE: {mae:.6f}")
    print(f"✓ Mean SSIM: {mean_ssim:.4f}")
    print(f"✓ Training sequences: {len(train_loader.dataset)}")
    print(f"✓ Test sequences: {len(test_loader.dataset)}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Image resolution: {img_height}x{img_width}")
    print(f"✓ Geographic coverage: Japan, Korea, Eastern China")
    print("\nFiles generated:")
    print("- convlstm_revin_predictions.png") # Changed filename
    print("- convlstm_revin_training.png") 
    print("- convlstm_revin_results.json")
    print("=" * 60)
    
    wandb.finish()
