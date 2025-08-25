import os
import sys
import random
import json
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import datetime

# Local imports
from stf_local.tsushima_preprocessing import (
    preprocess_tsushima_data,
    DATA_DIR,
    TARGET_VARIABLE,
    TSUSHIMA_LAT_MIN,
    TSUSHIMA_LAT_MAX,
    TSUSHIMA_LON_MIN,
    TSUSHIMA_LON_MAX,
    PREPROCESSED_DATA_PATH,
    SCALER_PATH,
    TRAIN_SPLIT_RATIO,
    MAX_FILES_TO_PROCESS,
)
from stf_local.spacetimeformer_model.nn.model import Spacetimeformer
from stf_local.lr_scheduler import WarmupReduceLROnPlateauScheduler

# Config
SEED = 42
LOOK_BACK_WINDOW = 30 
PREDICTION_HORIZON = 5  # Changed from 7
BATCH_SIZE = 4  # Keep as is since you have limited compute
NUM_EPOCHS = 25  # Changed from 10
LEARNING_RATE = 1e-5  # Keep as is
WEIGHT_DECAY = 1e-4  # Keep as is
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 8  # Changed from 10 for faster stopping

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Add after imports
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
    def forward(self, x, mode='norm'):
        # x shape: [B, L, N] or [B, H, N]
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + self.eps)
            x_normalized = (x - self.mean) / self.stdev
            if self.affine:
                x_normalized = x_normalized * self.affine_weight + self.affine_bias
            return x_normalized
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
            return x * self.stdev + self.mean
        else:
            raise NotImplementedError

class TsushimaSTFDataset(Dataset):
    """Dataset producing Spacetimeformer inputs with spatio-temporal embedding."""
    def __init__(self, data_np: np.ndarray, timestamps_tensor: torch.Tensor, look_back: int, horizon: int, is_train=False, augmenter=None):
        # data_np shape: (T, H, W)
        self.data = data_np.astype(np.float32)
        self.timestamps = timestamps_tensor.float()  # shape: (T, d_x)
        self.look_back = look_back
        self.horizon = horizon
        self.T, self.H, self.W = self.data.shape
        self.N = self.H * self.W
        self.is_train = is_train  # Add this line
        self.augmenter = augmenter  # Add this line

    def __len__(self):
        return self.T - self.look_back - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # Slices
        enc_slice = slice(idx, idx + self.look_back)
        dec_slice = slice(idx + self.look_back, idx + self.look_back + self.horizon)
        # enc_y: (L, N)
        enc_y = torch.from_numpy(self.data[enc_slice].reshape(self.look_back, self.N))
        # dec_y: zeros placeholder (H, N)
        dec_y = torch.zeros((self.horizon, self.N), dtype=torch.float32)
        # Targets: (H, N)
        y_target = torch.from_numpy(self.data[dec_slice].reshape(self.horizon, self.N))
        # enc_x/dec_x time features
        enc_x = self.timestamps[enc_slice]  # (L, d_x)
        dec_x = self.timestamps[dec_slice]  # (H, d_x)

        # Apply augmentation if in training mode and augmenter is available
        if self.is_train and self.augmenter is not None and random.random() < 0.7:
            enc_y = self.augmenter.augment(enc_y)
        
        return enc_x, enc_y, dec_x, dec_y, y_target

class BoundedL1Loss(nn.Module):
    def __init__(self, beta=0.2):
        super().__init__()
        self.beta = beta
        
    def forward(self, pred, target):
        # Basic L1 loss
        l1_loss = torch.abs(pred - target)
        
        # Additional penalty for out-of-bounds predictions
        bound_loss = torch.relu(-pred) + torch.relu(pred - 1)
        
        return l1_loss.mean() + self.beta * bound_loss.mean()
        
class TimeSeriesAugmenter:
    def __init__(self, noise_level=0.02, shift_level=0.05):
        self.noise_level = noise_level
        self.shift_level = shift_level
        
    def augment(self, x):
        # Add Gaussian noise
        noise = torch.randn_like(x) * self.noise_level * x.std()
        # Add small random shifts
        shift = torch.randn(1) * self.shift_level
        return x + noise + shift


def train():
    print("=== Tsushima Spacetimeformer Training ===")
    print(f"Device: {DEVICE}")
    set_seed(SEED)

    # Ensure preprocessed data exists or create it
    if not (os.path.exists(PREPROCESSED_DATA_PATH) and os.path.exists(SCALER_PATH)):
        print("Preprocessed data missing. Running preprocessing...")
        preprocess_tsushima_data(
            DATA_DIR,
            TARGET_VARIABLE,
            TSUSHIMA_LAT_MIN,
            TSUSHIMA_LAT_MAX,
            TSUSHIMA_LON_MIN,
            TSUSHIMA_LON_MAX,
            PREPROCESSED_DATA_PATH,
            SCALER_PATH,
            TRAIN_SPLIT_RATIO,
            max_files=MAX_FILES_TO_PROCESS,
        )

    # Load preprocessed normalized data and scaler
    norm_da = xr.open_dataarray(PREPROCESSED_DATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    data_np = norm_da.values  # (T, H, W)

    # After loading data (around line 107-108):
    print("Data statistics before scaling:")
    print(f"Raw data range: {data_np.min():.3f} to {data_np.max():.3f}")
    print(f"Raw data mean: {data_np.mean():.3f}")


    # Build timestamps tensor (year, month, day, hour)
    ts_dt = norm_da['time'].dt
    timestamps_tensor = torch.stack([
        torch.tensor(ts_dt.year.values, dtype=torch.float32),
        torch.tensor(ts_dt.month.values, dtype=torch.float32),
        torch.tensor(ts_dt.day.values, dtype=torch.float32),
        torch.tensor(ts_dt.hour.values, dtype=torch.float32)
    ], dim=-1)

    # Create augmenter
    augmenter = TimeSeriesAugmenter(noise_level=0.02, shift_level=0.03)

    # Load Data
    train_dataset_base = TsushimaSTFDataset(
    data_np, timestamps_tensor, LOOK_BACK_WINDOW, PREDICTION_HORIZON, 
    is_train=True, augmenter=augmenter
    )
    val_dataset_base = TsushimaSTFDataset(
        data_np, timestamps_tensor, LOOK_BACK_WINDOW, PREDICTION_HORIZON, 
        is_train=False, augmenter=None
    )

    total_samples = len(train_dataset_base)
    train_size = int(TRAIN_SPLIT_RATIO * total_samples)
    val_size = total_samples - train_size

    train_dataset = torch.utils.data.Subset(train_dataset_base, range(train_size))
    val_dataset = torch.utils.data.Subset(val_dataset_base, range(train_size, train_size + val_size))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    H, W = data_np.shape[1], data_np.shape[2]
    N = H * W
    d_x = timestamps_tensor.shape[-1]

    revin = RevIN(N).to(DEVICE)

    # Choose attn_time_windows that divides both encoder and decoder lengths
    from math import gcd
    enc_len = LOOK_BACK_WINDOW
    dec_len = PREDICTION_HORIZON
    # Use fixed window size instead of GCD
    attn_windows = 5  # Fixed value as recommended
    print(f"Using attn_time_windows={attn_windows}")

    # Model
    model = Spacetimeformer(
        d_yc=N,
        d_yt=N,
        d_x=d_x,
        start_token_len=0,
        attn_factor=5,
        d_model=128,  # Changed from 160
        d_queries_keys=32,  # Adjusted for 8 heads
        d_values=32,  # Adjusted for 8 heads
        n_heads=4,  # Changed from 4
        e_layers=3,
        d_layers=3,
        d_ff=512,  # Changed from 640
        initial_downsample_convs=0,
        intermediate_downsample_convs=0,
        dropout_emb=0.1,
        dropout_attn_out=0.0,
        dropout_attn_matrix=0.0,
        dropout_qkv=0.0,
        dropout_ff=0.2,
        pos_emb_type="t2v",  # Changed from "abs"
        global_self_attn="performer",
        local_self_attn="performer",
        global_cross_attn="performer",
        local_cross_attn="performer",
        activation="gelu",
        device=DEVICE,
        norm="batch",
        use_final_norm=True,
        embed_method="spatio-temporal",
        performer_attn_kernel="softmax",  # Changed from "relu"
        performer_redraw_interval=100,
        attn_time_windows=5,  # Changed to fixed value instead of using gcd
        use_shifted_time_windows=True,
        time_emb_dim=12,  # Changed from 6
        verbose=True,
        null_value=None,
        pad_value=None,
        max_seq_len=(LOOK_BACK_WINDOW + PREDICTION_HORIZON) * N,
        use_val=True,
        use_time=True,
        use_space=True,
        use_given=True,
        recon_mask_skip_all=0.9,  # Changed from 1.0
        recon_mask_max_seq_len=5,
        recon_mask_drop_seq=0.05,
        recon_mask_drop_standard=0.1,
        recon_mask_drop_full=0.02,
    ).to(DEVICE)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=WEIGHT_DECAY)
    # Warmup to LEARNING_RATE, then ReduceLROnPlateau
    warmup_steps = len(train_loader) * 5  # 3 epochs of warmup
    scheduler = WarmupReduceLROnPlateauScheduler(
        optimizer,
        init_lr=1e-8,
        peak_lr=LEARNING_RATE,
        warmup_steps=warmup_steps,
        patience=5,
        factor=0.5,
    )
    criterion = BoundedL1Loss(beta=0.2)  # Instead of nn.L1Loss()

    best_val = float('inf')
    best_epoch = -1
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for enc_x, enc_y, dec_x, dec_y, y_target in train_loader:
            enc_x = enc_x.to(DEVICE)
            enc_y = enc_y.to(DEVICE)
            dec_x = dec_x.to(DEVICE)
            dec_y = dec_y.to(DEVICE)
            y_target = y_target.to(DEVICE)

            # In training loop (modify lines 219-221):
            # Replace the duplicated code sections with a cleaner implementation
            optimizer.zero_grad()

            # First apply RevIN normalization
            enc_y_normalized = revin(enc_y, mode='norm')

            # Optional: Apply masking for self-supervised learning (30% of batches)
            if random.random() < 0.3:
                # Create random mask
                mask = torch.bernoulli(torch.ones_like(enc_y_normalized) * 0.15).bool()
                # Save original values
                original_enc_y = enc_y_normalized.clone()
                # Apply mask
                masked_enc_y = enc_y_normalized.clone()
                masked_enc_y[mask] = 0.0
                
                # Forward pass with masked input
                forecast_out, recon_out, _, _ = model(enc_x, masked_enc_y, dec_x, dec_y, output_attention=False)
                
                # Combined loss: forecasting + reconstruction
                forecast_loss = criterion(forecast_out, y_target)
                recon_loss = criterion(recon_out[mask], original_enc_y[mask]) 
                loss = forecast_loss + 0.3 * recon_loss
            else:
                # Regular forward pass
                forecast_out, _, _, _ = model(enc_x, enc_y_normalized, dec_x, dec_y, output_attention=False)
                # Denormalize output
                forecast_out = revin(forecast_out, mode='denorm')
                # Standard loss
                loss = criterion(forecast_out, y_target)

            # Continue with backward pass
            loss.backward()

            if epoch == 1 or epoch % 10 == 0:  # Check every 5 epochs
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient norm: {total_norm:.6f}")
            
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

            def detailed_loss_check(pred, target):
                with torch.no_grad():
                    invalid_pred = (pred < 0).sum().item() + (pred > 1).sum().item()
                    mean_pred = pred.mean().item()
                    mean_target = target.mean().item()
                    max_error = (pred - target).abs().max().item()
                    
                    print(f"Invalid predictions: {invalid_pred}")
                    print(f"Prediction mean: {mean_pred:.3f}, Target mean: {mean_target:.3f}")
                    print(f"Maximum error: {max_error:.3f}")

            # In training loop after loss calculation:
            if epoch <= 2:  # First two epochs only
                detailed_loss_check(forecast_out, y_target)
        
        train_loss /= max(1, len(train_loader))
        if epoch == 1 or epoch % 10 == 0:  # Print every 2 epochs
                print(f"\nEpoch {epoch} Statistics:")
                print(f"Training loss: {train_loss:.6f}")
                print(f"Learning rate: {scheduler.get_lr()[0]:.6e}")
                
                # Sample predictions from first batch
                with torch.no_grad():
                    sample_pred = forecast_out[0].cpu().numpy()
                    sample_true = y_target[0].cpu().numpy()
                    print(f"Sample prediction range: {sample_pred.min():.3f} to {sample_pred.max():.3f}")
                    print(f"Sample target range: {sample_true.min():.3f} to {sample_true.max():.3f}")
        # After calculating train_loss
        if epoch == 1:
            print("\nFirst Epoch Validation Check:")
            model.eval()
            with torch.no_grad():
                val_pred = forecast_out[0].cpu()  # Take first batch
                print(f"Validation Stats:")
                print(f"Pred range: {val_pred.min():.3f} to {val_pred.max():.3f}")
                print(f"Mean prediction: {val_pred.mean():.3f}")
        model.eval()
        val_loss = 0.0
        # Make sure RevIN is used in validation too
        with torch.no_grad():
            for enc_x, enc_y, dec_x, dec_y, y_target in val_loader:
                enc_x = enc_x.to(DEVICE)
                enc_y = enc_y.to(DEVICE)
                dec_x = dec_x.to(DEVICE)
                dec_y = dec_y.to(DEVICE)
                y_target = y_target.to(DEVICE)
                
                # Apply RevIN
                enc_y_normalized = revin(enc_y, mode='norm')
                forecast_out, _, _, _ = model(enc_x, enc_y_normalized, dec_x, dec_y, output_attention=False)
                # Denormalize output
                forecast_out = revin(forecast_out, mode='denorm')
                
                val_loss += criterion(forecast_out, y_target).item()
        val_loss /= max(1, len(val_loader))

        # Step plateau scheduler at epoch end
        scheduler.step(val_loss, is_end_epoch=True)
        current_lr = scheduler.get_lr()[0]
        print(f"Epoch {epoch:03d} | LR {current_lr:.2e} | Train {train_loss:.6f} | Val {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'tsushima_spacetimeformer_best.pt')
        elif epoch - best_epoch >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

    # Quick evaluation on validation set in original units (MAE day-1 average)
    model.load_state_dict(torch.load('tsushima_spacetimeformer_best.pt', map_location=DEVICE))
    model.eval()
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for enc_x, enc_y, dec_x, dec_y, y_target in val_loader:
            enc_x = enc_x.to(DEVICE)
            enc_y = enc_y.to(DEVICE)
            dec_x = dec_x.to(DEVICE)
            dec_y = dec_y.to(DEVICE)
            forecast_out, _, _, _ = model(enc_x, enc_y, dec_x, dec_y, output_attention=False)
            preds_all.append(forecast_out.cpu().numpy())
            trues_all.append(y_target.cpu().numpy())
    preds = np.concatenate(preds_all, axis=0)  # (B, H, N)
    trues = np.concatenate(trues_all, axis=0)

    # Inverse scale to original units
    preds_flat = preds.reshape(-1, 1)
    trues_flat = trues.reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds_flat).reshape(preds.shape)
    trues_inv = scaler.inverse_transform(trues_flat).reshape(trues.shape)

    # After inverse scaling (around line 276-277):
    print("Data statistics after inverse scaling:")
    print(f"Predictions range: {preds_inv.min():.3f} to {preds_inv.max():.3f}")
    print(f"Targets range: {trues_inv.min():.3f} to {trues_inv.max():.3f}")

    mae = mean_absolute_error(trues_inv.flatten(), preds_inv.flatten())
    print(f"Validation MAE (original units): {mae:.6f}")

    # Prepare files for DBSCAN GIF (use first forecast step per sample)
    try:
        H_map, W_map = H, W  # map dimensions
        # Take day-1 horizon for all samples
        preds_day0_maps = preds_inv[:, 0, :].reshape(-1, H_map, W_map)
        trues_day0_maps = trues_inv[:, 0, :].reshape(-1, H_map, W_map)
        # Build timestamps aligned with validation samples
        # val indices are consecutive from train_size .. train_size+val_size-1 and loader is not shuffled
        times = pd.to_datetime(norm_da.time.values)
        ts_list = []
        for k in range(trues_day0_maps.shape[0]):
            orig_idx = train_size + k
            pred_time_idx = orig_idx + LOOK_BACK_WINDOW
            if pred_time_idx < len(times):
                ts_list.append(times[pred_time_idx].strftime('%Y-%m-%d'))
            else:
                ts_list.append("N/A")
        np.save('spacetimeformer_predictions.npy', preds_day0_maps)
        np.save('spacetimeformer_targets.npy', trues_day0_maps)
        with open('spacetimeformer_timestamps.json', 'w') as f:
            json.dump(ts_list, f)
        print("Saved spacetimeformer_predictions.npy, spacetimeformer_targets.npy, spacetimeformer_timestamps.json for GIF generation")
    except Exception as e:
        print(f"Warning: failed to save spacetimeformer GIF inputs: {e}")

    # Save a small sample of predictions (first batch) for 7-day maps
    if len(val_loader) > 0:
        enc_x, enc_y, dec_x, dec_y, y_target = next(iter(val_loader))
        with torch.no_grad():
            forecast_out, _, _, _ = model(enc_x.to(DEVICE), enc_y.to(DEVICE), dec_x.to(DEVICE), dec_y.to(DEVICE))
        forecast_np = forecast_out.cpu().numpy()[0]  # (H, N)
        target_np = y_target.numpy()[0]
        # inverse scale and reshape to (H, H_map, W_map)
        forecast_maps = scaler.inverse_transform(forecast_np.reshape(-1, 1)).reshape(PREDICTION_HORIZON, H, W)
        target_maps = scaler.inverse_transform(target_np.reshape(-1, 1)).reshape(PREDICTION_HORIZON, H, W)
        np.save('tsushima_stf_pred_maps.npy', forecast_maps)
        np.save('tsushima_stf_true_maps.npy', target_maps)
        print("Saved example prediction maps to tsushima_stf_pred_maps.npy and tsushima_stf_true_maps.npy")

    # Create a visualization similar to japan_convlstm_predictions.png
    try:
        # Select first 4 and last 4 validation samples
        val_indices = list(val_dataset.indices) if hasattr(val_dataset, 'indices') else list(range(train_size, train_size + val_size))
        indices_to_plot = []
        if len(val_indices) >= 8:
            indices_to_plot.extend(val_indices[:4])
            indices_to_plot.extend(val_indices[-4:])
        else:
            indices_to_plot.extend(val_indices)

        # Global color limits in original units
        global_vmin_orig = scaler.inverse_transform(np.array([[data_np.min()]]))[0][0]
        global_vmax_orig = scaler.inverse_transform(np.array([[data_np.max()]]))[0][0]

        num_cols = len(indices_to_plot)
        if num_cols > 0:
            fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols + 1.5, 8))
            if num_cols == 1:
                axes = np.array(axes).reshape(2, 1)

            for col_idx, orig_idx in enumerate(indices_to_plot):
                # Fetch single sample directly from base dataset
                enc_x, enc_y, dec_x, dec_y, y_target = full_dataset[orig_idx]
                with torch.no_grad():
                    fo, _, _, _ = model(
                        enc_x.unsqueeze(0).to(DEVICE),
                        enc_y.unsqueeze(0).to(DEVICE),
                        dec_x.unsqueeze(0).to(DEVICE),
                        dec_y.unsqueeze(0).to(DEVICE),
                        output_attention=False,
                    )
                pred_flat = fo.squeeze(0).cpu().numpy()  # (H, N)
                true_flat = y_target.numpy()              # (H, N)
                # Use day 1 of the horizon for display
                pred_day0 = scaler.inverse_transform(pred_flat[0].reshape(-1, 1)).reshape(H, W)
                true_day0 = scaler.inverse_transform(true_flat[0].reshape(-1, 1)).reshape(H, W)
                # Compute per-sample MAE (day 1)
                sample_mae = mean_absolute_error(true_day0.flatten(), pred_day0.flatten())
                # Timestamp for predicted first day
                times = norm_da.time.values
                # predicted day index corresponds to orig_idx + LOOK_BACK_WINDOW
                pred_time_idx = orig_idx + LOOK_BACK_WINDOW
                if pred_time_idx < len(times):
                    date_str = pd.to_datetime(times[pred_time_idx]).strftime('%Y-%m-%d')
                else:
                    date_str = "N/A"

                ax_actual = axes[0, col_idx]
                im = ax_actual.imshow(true_day0, cmap='viridis', origin='lower', vmin=global_vmin_orig, vmax=global_vmax_orig)
                ax_actual.set_title(f"Actual\n{date_str}\nMAE: {sample_mae:.4f}", fontsize=10)
                ax_actual.axis('off')

                ax_pred = axes[1, col_idx]
                ax_pred.imshow(pred_day0, cmap='viridis', origin='lower', vmin=global_vmin_orig, vmax=global_vmax_orig)
                ax_pred.set_title("Predicted", fontsize=10)
                ax_pred.axis('off')

            fig.suptitle('Spacetimeformer Predictions vs Actual (Tsushima Region)\n(First and Last Validation Samples, Day 1 of Horizon)', fontsize=14)
            # Place a single colorbar at the far right side
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Concentration (Original Scale)')
            plt.tight_layout(rect=[0.02, 0.03, 0.9, 0.95])
            plt.savefig('tsushima_spacetimeformer_predictions.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved visualization to tsushima_spacetimeformer_predictions.png")
    except Exception as e:
        print(f"Warning: failed to create predictions figure: {e}")

if __name__ == '__main__':
    train()
