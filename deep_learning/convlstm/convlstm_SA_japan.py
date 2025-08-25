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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import json
import wandb
from utils.repro_eval import set_seed, evaluate_arrays

# Device and seed
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
set_seed(42)
print(f"Using device: {device}")

# --- Configuration ---
NC_FILES_PATH = Path("./data")
MAX_FILES_TO_PROCESS = 500
SEQ_LENGTH = 4
TEST_RATIO = 0.2

# Japan region (same as convlstm_japan.py)
JAPAN_SW_LAT, JAPAN_SW_LON = 25.35753, 118.85766
JAPAN_NE_LAT, JAPAN_NE_LON = 36.98134, 145.47117

# Output prefix for artifact names compatible with error_dbscan_gif.py
MODEL_PREFIX = 'convlstm_SA'

# --- ConvLSTM base ---
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
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
        combined = torch.cat([input_tensor, h_cur], dim=1)
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
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super().__init__()
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
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
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size[i], self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError("Stateful ConvLSTM not implemented")
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], [h, c])
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
        return [self.cell_list[i].init_hidden(batch_size, image_size) for i in range(self.num_layers)]

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# --- Spatial Attention (CBAM-like spatial attention) ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # input: concat(avg_pool, max_pool) over channel dim -> 2 x H x W
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


# --- ConvLSTM + Spatial Attention Model ---
class ConvLSTM_SA_Model(nn.Module):
    def __init__(self, seq_length, img_height, img_width,
                 input_channels=1, hidden_channels=[64, 64], kernel_size=(3, 3), num_layers=2):
        super().__init__()
        self.seq_length = seq_length
        self.img_height = img_height
        self.img_width = img_width
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.convlstm = ConvLSTM(
            input_dim=self.input_channels,
            hidden_dim=self.hidden_channels,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.decoder_conv = nn.Conv2d(in_channels=self.hidden_channels[-1], out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        layer_output_list, last_state_list = self.convlstm(x)
        last_h = last_state_list[-1][0]  # (B, C, H, W)
        attn_out = self.spatial_attn(last_h)
        out = self.decoder_conv(attn_out).squeeze(1)
        out = torch.tanh(out)
        out = (out + 1) / 2
        return out


# --- Data helpers ---
def extract_timestamps_from_filenames(nc_files_list):
    timestamps = []
    for file in nc_files_list:
        filename = Path(file).name
        date_part = filename.split('.')[2][1:9]
        timestamp = datetime.strptime(date_part, '%Y%m%d')
        timestamps.append(timestamp)
    return timestamps


def lat_lon_to_indices(lat, lon, lats, lons):
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx


def process_and_normalize_image(nc_file):
    ds = xr.open_dataset(nc_file)
    data = ds['mp_concentration']
    data_array_2d = data.values.squeeze()

    lats = None
    lons = None
    for lat_name in ['lat', 'latitude', 'y', 'lat_1', 'lat_2']:
        if lat_name in ds.variables:
            lats = ds[lat_name].values
            break
    for lon_name in ['lon', 'longitude', 'x', 'lon_1', 'lon_2']:
        if lon_name in ds.variables:
            lons = ds[lon_name].values
            break

    if lats is not None and lons is not None:
        sw_lat_idx, sw_lon_idx = lat_lon_to_indices(JAPAN_SW_LAT, JAPAN_SW_LON, lats, lons)
        ne_lat_idx, ne_lon_idx = lat_lon_to_indices(JAPAN_NE_LAT, JAPAN_NE_LON, lats, lons)
        lat_start = min(sw_lat_idx, ne_lat_idx)
        lat_end = max(sw_lat_idx, ne_lat_idx)
        lon_start = min(sw_lon_idx, ne_lon_idx)
        lon_end = max(sw_lon_idx, ne_lon_idx)
        cropped = data_array_2d[lat_start:lat_end, lon_start:lon_end]
    else:
        print(f"Warning: No coordinate information found in {nc_file}, using full dataset.")
        cropped = data_array_2d

    data_min = np.nanmin(cropped)
    data_max = np.nanmax(cropped)
    if data_max > data_min:
        normalized = (cropped - data_min) / (data_max - data_min)
    else:
        normalized = np.zeros_like(cropped)
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized


def load_and_preprocess_data():
    nc_files_list = sorted(NC_FILES_PATH.glob("cyg.ddmi*.nc"))
    print(f"Found {len(nc_files_list)} NetCDF files.")
    nc_files_limited = nc_files_list[:MAX_FILES_TO_PROCESS]
    print(f"Processing {len(nc_files_limited)} files (limited from {len(nc_files_list)} total files)")

    processed = []
    first_image = process_and_normalize_image(nc_files_limited[0])
    target_height, target_width = 128, 128
    processed.append(resize(first_image, (target_height, target_width), preserve_range=True))

    for i, nc_file in enumerate(nc_files_limited[1:]):
        if (i + 1) % 100 == 0:
            print(f"Processed {i+2}/{len(nc_files_limited)} files...")
        img = process_and_normalize_image(nc_file)
        processed.append(resize(img, (target_height, target_width), preserve_range=True))

    data = np.array(processed)
    print(f"Final preprocessed data shape: {data.shape}")
    return data, extract_timestamps_from_filenames(nc_files_limited)


def create_convlstm_sequences(data, seq_length):
    sequences = []
    targets = []
    num_samples, img_h, img_w = data.shape
    for i in range(num_samples - seq_length):
        seq = data[i: i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq[:, np.newaxis, :, :])
        targets.append(target)
    X = np.array(sequences)
    Y = np.array(targets)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y, img_h, img_w


def temporal_train_test_split_proper(data, seq_length, test_ratio=0.2):
    total_files = len(data)
    test_files_needed = int(total_files * test_ratio)
    gap_needed = seq_length
    min_train_files = seq_length + 1
    min_files_needed = min_train_files + gap_needed + test_files_needed + seq_length
    if total_files < min_files_needed:
        print(f"WARNING: Only {total_files} files available, need {min_files_needed} for proper split")
        test_files_needed = max(1, (total_files - min_train_files - gap_needed - seq_length) // 2)
    train_end = total_files - test_files_needed - gap_needed - seq_length
    test_start = train_end + gap_needed
    train_data = data[:train_end]
    test_data = data[test_start:]
    print(f"Train files: 0..{train_end-1}, gap: {train_end}..{test_start-1}, test: {test_start}..{total_files-1}")
    return train_data, test_data, train_end, test_start


# --- Training / Evaluation ---
def improved_contrast_loss(pred, target, low_threshold=0.2, high_threshold=0.6,
                          low_weight=2.0, high_weight=2.0, contrast_weight=1.0):
    mae = torch.mean(torch.abs(pred - target))
    low_mask = (target < low_threshold).float()
    high_mask = (target > high_threshold).float()
    mid_mask = 1.0 - low_mask - high_mask
    weighted_mae = torch.mean(torch.abs(pred - target) * (1.0 + low_weight * low_mask + high_weight * high_mask + 0.5 * mid_mask))
    pred_std = torch.std(pred)
    target_std = torch.std(target)
    contrast_loss = torch.abs(pred_std - target_std)
    pred_grad_x = torch.abs(pred[:, :-1, :] - pred[:, 1:, :])
    target_grad_x = torch.abs(target[:, :-1, :] - target[:, 1:, :])
    pred_grad_y = torch.abs(pred[:, :, :-1] - pred[:, :, 1:])
    target_grad_y = torch.abs(target[:, :, :-1] - target[:, :, 1:])
    edge_loss = torch.mean(torch.abs(pred_grad_x - target_grad_x)) + torch.mean(torch.abs(pred_grad_y - target_grad_y))
    return weighted_mae + contrast_weight * contrast_loss + 0.1 * edge_loss


def train_model(model, train_loader, val_loader, epochs=10, learning_rate=1e-3):
    print(f"\nTraining {MODEL_PREFIX} for {epochs} epochs...")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        train_loss = 0.0
        for batch_idx, (bx, by) in enumerate(train_loader):
            if batch_idx % 5 == 0:
                print(f"  Training batch {batch_idx+1}/{len(train_loader)}")
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = improved_contrast_loss(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                out = model(bx)
                loss = improved_contrast_loss(out, by)
                val_loss += loss.item()
        history['train_loss'].append(train_loss / max(1, len(train_loader)))
        history['val_loss'].append(val_loss / max(1, len(val_loader)))
        print(f"  Train {history['train_loss'][-1]:.6f} | Val {history['val_loss'][-1]:.6f}")
    return history


def evaluate_and_save(model, test_loader, timestamps, test_start_idx):
    print("\nEvaluating...")
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            out = model(bx)
            preds.extend(out.cpu().numpy())
            trues.extend(by.cpu().numpy())
    preds = np.array(preds)
    trues = np.array(trues)

    overall_mae = mean_absolute_error(trues.flatten(), preds.flatten())
    ssim_scores = []
    for i in range(len(preds)):
        p = preds[i]
        t = trues[i]
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
        ssim_scores.append(ssim(t_norm, p_norm, data_range=1.0))
    mean_ssim = float(np.mean(ssim_scores))

    individual_maes = np.array([mean_absolute_error(trues[i].flatten(), preds[i].flatten()) for i in range(len(preds))])

    test_ts = timestamps[test_start_idx: test_start_idx + len(trues)]

    # Save artifacts for GIFs
    np.save(f'{MODEL_PREFIX}_predictions.npy', preds)
    np.save(f'{MODEL_PREFIX}_targets.npy', trues)
    try:
        err_maps = np.abs(preds - trues)
        np.save(f'{MODEL_PREFIX}_error_maps.npy', err_maps)
    except Exception as e:
        print(f"Warning: failed to save error maps: {e}")
    np.save(f'{MODEL_PREFIX}_individual_maes.npy', individual_maes)
    with open(f'{MODEL_PREFIX}_timestamps.json', 'w') as f:
        json.dump([ts.strftime('%Y-%m-%d') for ts in test_ts], f)

    # Create a 2xN figure of first 4 and last 4
    num_cols = 8 if len(preds) >= 8 else len(preds)
    indices = list(range(min(4, len(preds))))
    if len(preds) >= 8:
        indices += list(range(len(preds) - 4, len(preds)))
    indices = sorted(list(set(indices)))

    fig, axes = plt.subplots(2, len(indices), figsize=(4 * len(indices), 10))
    fig.suptitle(f'{MODEL_PREFIX} Predictions vs Actual (Japan Region)', fontsize=16)
    if len(indices) == 1:
        axes = axes.reshape(2, 1)
    for col_idx, i in enumerate(indices):
        t_disp = np.flipud(trues[i])
        p_disp = np.flipud(preds[i])
        axes[0, col_idx].imshow(t_disp, cmap='viridis', aspect='auto')
        axes[0, col_idx].set_title(f'Actual\n{test_ts[i].strftime("%Y-%m-%d")}\nMAE: {individual_maes[i]:.4f}')
        axes[0, col_idx].axis('off')
        axes[1, col_idx].imshow(p_disp, cmap='viridis', aspect='auto')
        axes[1, col_idx].set_title(f'Predicted\nMAE: {individual_maes[i]:.4f}')
        axes[1, col_idx].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig(f'japan_{MODEL_PREFIX}_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Overall Test MAE: {overall_mae:.6f}")
    print(f"Overall Mean SSIM: {mean_ssim:.4f}")
    return overall_mae, mean_ssim


if __name__ == "__main__":
    print("=" * 60)
    print("JAPAN REGION ConvLSTM + Spatial Attention (SA)")
    print("=" * 60)

    all_images, timestamps_full = load_and_preprocess_data()
    train_imgs, test_imgs, train_end_idx, test_start_idx = temporal_train_test_split_proper(all_images, SEQ_LENGTH, TEST_RATIO)

    X_train, y_train, img_h, img_w = create_convlstm_sequences(train_imgs, SEQ_LENGTH)
    X_test, y_test, _, _ = create_convlstm_sequences(test_imgs, SEQ_LENGTH)

    # Validation split from training (temporal)
    val_split_idx = int(len(X_train) * 0.8)
    X_train_final = X_train[:val_split_idx]
    y_train_final = y_train[:val_split_idx]
    X_val = X_train[val_split_idx:]
    y_val = y_train[val_split_idx:]

    # Tensors and loaders
    X_train_tensor = torch.FloatTensor(X_train_final)
    y_train_tensor = torch.FloatTensor(y_train_final)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    batch_size = 8
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # wandb
    wandb.init(
        project="mp-prediction-convlstm-SA",
        config={
            "model_type": MODEL_PREFIX,
            "framework": "PyTorch",
            "sequence_length": SEQ_LENGTH,
            "learning_rate": 0.001,
            "batch_size": batch_size,
            "epochs": 10,
            "image_resolution": f"{img_h}x{img_w}",
            "input_channels": 1,
            "hidden_channels": [64, 64],
            "kernel_size": (3, 3),
            "num_layers": 2,
            "loss_function": "improved_contrast_loss",
            "geographic_coverage": "Japan_Korea_EasternChina",
            "data_files_processed": MAX_FILES_TO_PROCESS,
        },
    )

    model = ConvLSTM_SA_Model(
        seq_length=SEQ_LENGTH,
        img_height=img_h,
        img_width=img_w,
        input_channels=1,
        hidden_channels=[64, 64],
        kernel_size=(3, 3),
        num_layers=2,
    )

    print("\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    history = train_model(model, train_loader, val_loader, epochs=10, learning_rate=1e-3)
    mae, mean_ssim = evaluate_and_save(model, test_loader, timestamps_full, test_start_idx)
    # Overall metrics
    preds = np.load(f'{MODEL_PREFIX}_predictions.npy')
    tgts = np.load(f'{MODEL_PREFIX}_targets.npy')
    metrics = evaluate_arrays(tgts, preds, tol=0.10)
    print(f"Metrics ({MODEL_PREFIX}): R2={metrics['r2']:.4f} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} Acc@10%={metrics['acc_within_10pct']:.3f}")

    # Log
    wandb.log({
        "test_mae": mae,
        "test_ssim": mean_ssim,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "test_r2": metrics['r2'],
        "test_rmse": metrics['rmse'],
        "test_acc10": metrics['acc_within_10pct'],
    })

    # Training history plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('ConvLSTM+SA Training')
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
    plt.savefig(f'japan_{MODEL_PREFIX}_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save summary
    results = {
        'model_type': MODEL_PREFIX,
        'framework': 'PyTorch',
        'device': str(device),
        'sequence_length': SEQ_LENGTH,
        'image_resolution': f"{img_h}x{img_w}",
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
            'convlstm_num_layers': 2,
            'spatial_attention': 'CBAM-like spatial mask'
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
    with open(f'japan_{MODEL_PREFIX}_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("JAPAN REGION ConvLSTM + Spatial Attention COMPLETED")
    print("=" * 60)
    print(f"✓ Test MAE: {mae:.6f}")
    print(f"✓ Mean SSIM: {mean_ssim:.4f}")
    print(f"✓ Training sequences: {len(train_loader.dataset)}")
    print(f"✓ Test sequences: {len(test_loader.dataset)}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Image resolution: {img_h}x{img_w}")
    print(f"✓ Files saved with prefix '{MODEL_PREFIX}' for GIF generation")

    wandb.finish()
