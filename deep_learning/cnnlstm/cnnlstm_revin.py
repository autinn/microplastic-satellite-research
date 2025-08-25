import sys
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
device = torch.device('mps')
set_seed(42)
print(f"Using device: {device}")

# --- Configuration ---
# Set the path to your NetCDF files
# Make sure your 'data' folder is in the same directory as this script
NC_FILES_PATH = Path("./data")
# Max files to process (keep consistent with other models)
MAX_FILES_TO_PROCESS = 500 
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
        dim2reduce = 1 
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()


# --- Data Loading and Preprocessing Functions ---

def extract_timestamps_from_filenames(nc_files_list):
    """Extract timestamps from NetCDF filenames."""
    timestamps = []
    for file in nc_files_list:
        filename = Path(file).name
        date_part = filename.split('.')[2][1:9]
        timestamp = datetime.strptime(date_part, '%Y%m%d')
        timestamps.append(timestamp)
    return timestamps

def downsample_data(data_array, target_height=128, target_width=128):
    """Downsample to higher resolution optimized for Japan region."""
    if data_array.ndim == 2:
        return resize(data_array, (target_height, target_width), preserve_range=True)
    elif data_array.ndim == 3:
        return np.array([resize(img, (target_height, target_width), preserve_range=True) for img in data_array])
    else:
        raise ValueError(f"Unexpected number of dimensions: {data_array.ndim}")

def lat_lon_to_indices(lat, lon, lats, lons):
    """Convert lat/lon coordinates to grid indices"""
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx

def process_and_normalize_image(nc_file):
    """Convert a netCDF file to a Japan region cropped normalized array."""
    ds = xr.open_dataset(nc_file)
    data = ds['mp_concentration']
    data_array_2d = data.values.squeeze()
    
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
        japan_sw_lat, japan_sw_lon = JAPAN_SW_LAT, JAPAN_SW_LON
        japan_ne_lat, japan_ne_lon = JAPAN_NE_LAT, JAPAN_NE_LON
        
        japan_sw_lat_idx, japan_sw_lon_idx = lat_lon_to_indices(japan_sw_lat, japan_sw_lon, lats, lons)
        japan_ne_lat_idx, japan_ne_lon_idx = lat_lon_to_indices(japan_ne_lat, japan_ne_lon, lats, lons)
        
        japan_lat_start = min(japan_sw_lat_idx, japan_ne_lat_idx)
        japan_lat_end = max(japan_sw_lat_idx, japan_ne_lat_idx)
        japan_lon_start = min(japan_sw_lon_idx, japan_ne_lon_idx)
        japan_lon_end = max(japan_sw_lon_idx, japan_ne_lon_idx)
        
        cropped_data = data_array_2d[japan_lat_start:japan_lat_end, japan_lon_start:japan_lon_end]
        
        # Check if nc_file is the first file in the *limited* list for printing
        # This global nc_files is defined later in __main__, so we need to pass it or re-sort
        # For simplicity, we'll just print for the first file in the overall sorted list
        if nc_file == sorted(Path("./data").glob("cyg.ddmi*.nc"))[0]: # Use Path("./data") directly
            print(f"Japan region cropped to: lat[{japan_lat_start}:{japan_lat_end}], lon[{japan_lon_start}:{japan_lon_end}]")
            print(f"Japan region shape: {cropped_data.shape}")
    else:
        print("Warning: No coordinate information found, using full dataset")
        cropped_data = data_array_2d
    
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
    nc_files_list = sorted(Path("./data").glob("cyg.ddmi*.nc")) # Use Path("./data") directly
    print(f"Found {len(nc_files_list)} NetCDF files.")
    
    nc_files_limited = nc_files_list[:MAX_FILES_TO_PROCESS]
    print(f"Processing {len(nc_files_limited)} files (limited from {len(nc_files_list)} total files)")
    
    processed_images = []
    
    first_image = process_and_normalize_image(nc_files_limited[0])
    target_height, target_width = 128, 128 
    first_image_resized = downsample_data(first_image, target_height, target_width)
    processed_images.append(first_image_resized)

    print(f"Target image resolution set to: {target_height}x{target_width}")
    
    for i, nc_file in enumerate(nc_files_limited[1:]):
        if (i + 1) % 100 == 0:
            print(f"Processed {i+2}/{len(nc_files_limited)} files...")
        
        img = process_and_normalize_image(nc_file)
        img_resized = downsample_data(img, target_height, target_width)
        processed_images.append(img_resized)
            
    data = np.array(processed_images)
    print(f"Final preprocessed data shape: {data.shape}")
    return data, extract_timestamps_from_filenames(nc_files_limited)


def temporal_train_test_split_proper(data, seq_length, test_ratio=0.2):
    """
    Properly split time series data to avoid leakage.
    """
    total_files = len(data)
    
    test_files_needed = int(total_files * test_ratio)
    gap_needed = seq_length  
    
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
    
    train_data = data[:train_end]
    test_data = data[test_start:]
    
    return train_data, test_data, train_end, test_start

def create_sequences_from_data(data, seq_length):
    """Create sequences from data array."""
    sequences = []
    target_images = []
    
    num_samples, img_height, img_width = data.shape
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length] 
        target_image = data[i + seq_length]
        
        # Add channel dimension: (seq_length, H, W) -> (seq_length, 1, H, W)
        seq_with_channel = seq[:, np.newaxis, :, :]
        
        sequences.append(seq_with_channel)
        target_images.append(target_image)
        
    X = np.array(sequences) # (num_sequences, seq_length, 1, H, W)
    Y = np.array(target_images) # (num_sequences, H, W)
    
    print(f"Created CNN-LSTM sequences:")
    print(f"X (context) shape: {X.shape}")
    print(f"Y (target) shape: {Y.shape}")
    
    return X, Y


class EfficientCNNLSTMModel(nn.Module):
    """Efficient CNN-LSTM model that won't get stuck during training."""
    
    def __init__(self, seq_length, img_height, img_width):
        super(EfficientCNNLSTMModel, self).__init__()
        
        self.conv1_filters = 32
        self.conv2_filters = 64
        self.lstm_units = 128
        self.dropout_rate = 0.1
        
        self.seq_length = seq_length
        self.img_height = img_height
        self.img_width = img_width
        
        # NEW: RevIN layer
        self.revin = RevIN(num_features=1 * self.img_height * self.img_width) # C=1 for microplastic images

        # Input to conv1 is (B, C=1, H, W)
        self.conv1 = nn.Conv2d(1, self.conv1_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.conv1_filters, self.conv2_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Correct conv_output_width calculation based on actual img_width
        conv_output_height = img_height // 4
        conv_output_width = img_width // 4
        conv_output_size = self.conv2_filters * conv_output_height * conv_output_width
        
        self.lstm = nn.LSTM(conv_output_size, self.lstm_units, batch_first=True)
        
        self.decoder = nn.Linear(self.lstm_units, img_height * img_width)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, 1, height, width)
        batch_size, seq_length, channels, height, width = x.shape
        
        # Apply RevIN normalization
        # Reshape for RevIN: (B, T, C*H*W)
        x_flattened_for_revin = x.reshape(batch_size, seq_length, channels * height * width)
        x_normalized = self.revin(x_flattened_for_revin, 'norm')
        # Reshape back to (B, T, C, H, W) for CNN
        x_normalized = x_normalized.reshape(batch_size, seq_length, channels, height, width)

        cnn_outputs = []
        for t in range(seq_length):
            # x_t shape: (batch_size, channels, height, width)
            # This slice directly gives (B, C, H, W), which is what Conv2d expects.
            # No unsqueeze(-1) or permute needed here.
            x_t = x_normalized[:, t, :, :, :] 
            
            conv1_out = F.relu(self.conv1(x_t))
            conv1_out = self.pool(conv1_out)
            conv1_out = self.dropout(conv1_out)
            
            conv2_out = F.relu(self.conv2(conv1_out))
            conv2_out = self.pool(conv2_out)
            conv2_out = self.dropout(conv2_out)
            
            conv2_out_flat = conv2_out.view(batch_size, -1)
            cnn_outputs.append(conv2_out_flat)
        
        cnn_outputs = torch.stack(cnn_outputs, dim=1)
        
        lstm_out, _ = self.lstm(cnn_outputs)
        
        last_output = lstm_out[:, -1, :]
        
        output = self.decoder(last_output)
        output = output.view(batch_size, height, width)
        
        # Apply RevIN denormalization
        output_flattened_for_revin = output.reshape(batch_size, 1, height * width)
        output_denormalized = self.revin(output_flattened_for_revin, 'denorm')
        output_denormalized = output_denormalized.reshape(batch_size, height, width)

        # Apply final activation and scaling
        output_final = torch.tanh(output_denormalized)
        output_final = (output_final + 1) / 2
        
        return output_final

def improved_contrast_loss(pred, target, low_threshold=0.2, high_threshold=0.6, 
                          low_weight=2.0, high_weight=2.0, contrast_weight=1.0):
    """
    Improved loss function that balances high/low concentrations and encourages contrast.
    """
    mae = torch.mean(torch.abs(pred - target))
    
    low_concentration_mask = (target < low_threshold).float()
    high_concentration_mask = (target > high_threshold).float()
    mid_concentration_mask = 1.0 - low_concentration_mask - high_concentration_mask
    
    weighted_mae = torch.mean(
        torch.abs(pred - target) * (
            1.0 +  
            low_weight * low_concentration_mask +      
            high_weight * high_concentration_mask +    
            0.5 * mid_concentration_mask               
        )
    )
    
    pred_std = torch.std(pred)
    target_std = torch.std(target)
    contrast_loss = torch.abs(pred_std - target_std)
    
    pred_grad_x = torch.abs(pred[:, :-1, :] - pred[:, 1:, :])
    target_grad_x = torch.abs(target[:, :-1, :] - target[:, 1:, :])
    pred_grad_y = torch.abs(pred[:, :, :-1] - pred[:, :, 1:])
    target_grad_y = torch.abs(target[:, :, :-1] - target[:, :, 1:])
    
    edge_loss = torch.mean(torch.abs(pred_grad_x - target_grad_x)) + \
                torch.mean(torch.abs(pred_grad_y - target_grad_y))
    
    total_loss = weighted_mae + contrast_weight * contrast_loss + 0.1 * edge_loss
    
    return total_loss

def train_efficient_model_pytorch(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """Train the model efficiently with progress monitoring."""
    print(f"\nStarting efficient model training for {epochs} epochs...")
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if batch_idx % 5 == 0:
                print(f"  Training batch {batch_idx+1}/{len(train_loader)}")
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = improved_contrast_loss(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
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

    np.save('cnnlstm_revin_predictions.npy', predictions) 
    np.save('cnnlstm_revin_targets.npy', targets) 
    # Save per-frame absolute error maps for downstream visualization
    try:
        error_maps = np.abs(predictions - targets)
        np.save('cnnlstm_revin_error_maps.npy', error_maps)
    except Exception as e:
        print(f"Warning: failed to save cnnlstm_revin error maps: {e}")
    np.save('cnnlstm_revin_individual_maes.npy', individual_maes)
    
    with open('cnnlstm_revin_timestamps.json', 'w') as f:
        json.dump([ts.strftime("%Y-%m-%d") for ts in test_timestamps_aligned], f)
    
    print("\nSaved CNN-LSTM (with RevIN) predictions, targets, MAEs, and timestamps for GIF generation.")

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
    
    model_type = "CNN-LSTM + RevIN" 
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
    plt.savefig('japan_cnnlstm_revin_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return overall_mae, mean_ssim, predictions

# --- Main Execution ---

if __name__ == "__main__":
    print("=" * 60)
    print("JAPAN REGION CNN-LSTM FOR MICROPLASTIC CONCENTRATION")
    print("=" * 60)
    
    # --- Configuration (Local to main execution) ---
    NC_FILES_PATH = Path("./data")
    MAX_FILES_TO_PROCESS = 500 
    SEQ_LENGTH = 4 
    TEST_RATIO = 0.2 
    JAPAN_SW_LAT, JAPAN_SW_LON = 25.35753, 118.85766
    JAPAN_NE_LAT, JAPAN_NE_LON = 36.98134, 145.47117
    
    print(f"Processing files from: {NC_FILES_PATH}")

    # --- FIX: Define nc_files here for data loading functions ---
    nc_files = sorted(NC_FILES_PATH.glob("cyg.ddmi*.nc")) 
    print(f"Found {len(nc_files)} NetCDF files.")
    # --- END FIX ---

    # Extract data and timestamps
    # load_and_preprocess_data returns (data_array, timestamps_list)
    data, timestamps_full = load_and_preprocess_data() 
    print(f"Data spans from {timestamps_full[0]} to {timestamps_full[-1]}")
    
    # Define img_height and img_width here
    img_height, img_width = data[0].shape[:2]
    print(f"Image dimensions: {img_height}x{img_width}")

    print(f"\nUsing sequence length: {SEQ_LENGTH}")
    
    # Proper temporal split of the full image data
    train_data_images, test_data_images, train_end_idx, test_start_idx = temporal_train_test_split_proper(
        data, SEQ_LENGTH, TEST_RATIO
    )
    
    # Create sequences separately for train and test
    X_train, y_train = create_sequences_from_data(train_data_images, SEQ_LENGTH)
    X_test, y_test = create_sequences_from_data(test_data_images, SEQ_LENGTH)
    
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
    batch_size = 8
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize wandb
    wandb.init(
        project="mp-prediction-cnnlstm-revin", # Unique project name for RevIN
        config={
            "model_type": "CNN-LSTM + RevIN", # Updated model type
            "framework": "PyTorch",
            "sequence_length": SEQ_LENGTH, # Using locally defined SEQ_LENGTH
            "learning_rate": 0.001,
            "batch_size": batch_size,
            "epochs": 10,
            "image_resolution": f"{img_height}x{img_width}", # Now img_height/width are defined
            "aspect_ratio": "1:1_width_height", # Updated based on 128x128 images
            "loss_function": "improved_contrast_loss",
            "high_concentration_threshold": 0.4,
            "high_concentration_weight": 5.0,
            "cnn_layers": 2,
            "lstm_layers": 1,
            "conv_filters": [32, 64],
            "lstm_units": 128,
            "dropout_rate": 0.1,
            "simplified_architecture": True,
            "revin_enabled": True, # New config field to indicate RevIN is used
            "geographic_fixes": {
                "vertical_flip_correction": True,
                "proper_aspect_ratio": "1:1", # Updated based on 128x128 images
                "geographic_coverage": "Japan_Korea_EasternChina"
            },
            "data_files_processed": MAX_FILES_TO_PROCESS # Log max files processed
        }
    )
    
    # Create model
    model = EfficientCNNLSTMModel(SEQ_LENGTH, img_height, img_width) 
    
    print("\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = train_efficient_model_pytorch(model, train_loader, val_loader, epochs=10)
    
    # Evaluate model
    # Pass timestamps_full as it contains all original timestamps for correct indexing
    mae, mean_ssim, predictions = evaluate_model(model, test_loader, timestamps_full, test_start_idx)
    metrics = evaluate_arrays(np.load('cnnlstm_revin_targets.npy'), predictions, tol=0.10)
    print(f"Metrics (cnnlstm_revin): R2={metrics['r2']:.4f} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} Acc@10%={metrics['acc_within_10pct']:.3f}")
    
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
    plt.plot(history['train_loss'], label='Training MAE')
    plt.plot(history['val_loss'], label='Validation MAE')
    plt.title('Efficient Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'], 'o-', label='Train')
    plt.plot(range(1, len(history['val_loss'])+1), history['val_loss'], 's-', label='Validation')
    plt.title('Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss')
    plt.legend()
    
    plt.tight_layout()
    # Save training plot with RevIN specific filename
    plt.savefig('japan_cnnlstm_revin_training.png', dpi=150, bbox_inches='tight') 
    plt.close()
    
    # Save simplified results
    results = {
        'model_type': 'Efficient Japan Region CNN-LSTM with Geographic Corrections + RevIN',
        'framework': 'PyTorch',
        'device': str(device),
        'sequence_length': SEQ_LENGTH, # Using locally defined SEQ_LENGTH
        'image_resolution': f"{img_height}x{img_width}", # Now img_height/width are defined
        'training_samples': len(X_train_final),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'test_mae': mae,
        'test_ssim': mean_ssim,
        'model_parameters': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'conv1_filters': model.conv1_filters,
            'conv2_filters': model.conv2_filters,
            'lstm_units': model.lstm_units,
            'dropout_rate': model.dropout_rate,
            "simplified_architecture": True,
            "revin_enabled": True # Log that RevIN was enabled
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
            'coverage': 'Japan, Korea, and Eastern China maritime regions',
            'fixes_applied': {
                'vertical_flip': True,
                'aspect_ratio': '1:1 (width:height)', # Updated description
                'simplified_model': True
            }
        }
    }
    
    # Save results to a unique JSON file for RevIN version
    with open('japan_cnnlstm_revin_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EFFICIENT JAPAN REGION CNN-LSTM + REV-IN TRAINING COMPLETED")
    print("=" * 60)
    print(f"✓ Test MAE: {mae:.6f}")
    print(f"✓ Mean SSIM: {mean_ssim:.4f}")
    print(f"✓ Training files: {train_end_idx}")
    print(f"✓ Test files: {len(test_dataset)}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Image resolution: {img_height}x{img_width}") # Now img_height/width are defined
    print(f"✓ Geographic coverage: Japan, Korea, Eastern China")
    print(f"✓ Orientation: Corrected (vertical flip applied)")
    print("\nFiles generated:")
    print("- japan_cnnlstm_revin_predictions.png")
    print("- japan_cnnlstm_revin_training.png")
    print("- japan_cnnlstm_revin_results.json")
    print("=" * 60)
    
    # Finish wandb run
    wandb.finish()
