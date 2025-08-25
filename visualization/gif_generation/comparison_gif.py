import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio # Use imageio.v2 for newer versions
import json
from datetime import datetime
import os

# --- Configuration for GIF Generation ---
GIF_OUTPUT_FILENAME = 'microplastic_prediction_3x2_comparison.gif' # New filename for 3x2 layout
FRAMES_DIR = 'gif_frames_3x2' # New directory for frames
FPS = 3 # Frames per second for the GIF
MAX_FRAMES = 100 # Limit the number of frames to avoid very large GIFs (e.g., first 100 test samples)

# --- Load Data from Saved Files ---
def load_model_data(model_prefix):
    """Loads predictions, targets, MAEs, and timestamps for a given model."""
    try:
        predictions = np.load(f'{model_prefix}_predictions.npy')
        targets = np.load(f'{model_prefix}_targets.npy')
        individual_maes = np.load(f'{model_prefix}_individual_maes.npy')
        
        with open(f'{model_prefix}_timestamps.json', 'r') as f:
            timestamps_str = json.load(f)
            timestamps = [datetime.strptime(ts, "%Y-%m-%d") for ts in timestamps_str]
        
        print(f"Successfully loaded data for {model_prefix.upper()}.")
        return predictions, targets, individual_maes, timestamps
    except FileNotFoundError as e:
        print(f"Error: Could not find data files for {model_prefix.upper()}. Please ensure you have run the respective model scripts first.")
        print(e)
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading data for {model_prefix.upper()}: {e}")
        return None, None, None, None

def create_gif_frames(ground_truth_data, cnnlstm_std_preds, convlstm_std_preds, cnnlstm_revin_preds, convlstm_revin_preds_data, timestamps):
    """
    Creates individual frames (PNG images) for the GIF animation.
    Each frame shows comparisons in a 3x2 layout:
    Row 1: Ground Truth | Ground Truth
    Row 2: ConvLSTM     | ConvLSTM + RevIN
    Row 3: CNN-LSTM     | CNN-LSTM + RevIN
    """
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)

    num_frames = min(MAX_FRAMES, len(ground_truth_data))
    print(f"\nGenerating {num_frames} GIF frames...")

    # Determine global min/max for consistent colormap across all images
    all_data_values = np.concatenate([ground_truth_data.flatten(), 
                                      cnnlstm_std_preds.flatten(), 
                                      convlstm_std_preds.flatten(),
                                      cnnlstm_revin_preds.flatten(), # Include CNN-LSTM + RevIN
                                      convlstm_revin_preds_data.flatten()]) # Include ConvLSTM + RevIN
    vmin_global = np.min(all_data_values)
    vmax_global = np.max(all_data_values)

    # Ensure vmin/vmax are not identical for colormap if data is flat
    if vmin_global == vmax_global:
        vmin_global = 0.0
        vmax_global = 1.0 # Assuming normalized 0-1 range

    for i in range(num_frames):
        fig, axes = plt.subplots(3, 2, figsize=(12, 18)) # 3 rows, 2 columns, adjusted figsize
        
        current_gt = np.flipud(ground_truth_data[i])
        current_cnnlstm_std_pred = np.flipud(cnnlstm_std_preds[i])
        current_convlstm_std_pred = np.flipud(convlstm_std_preds[i])
        current_cnnlstm_revin_pred = np.flipud(cnnlstm_revin_preds[i])
        current_convlstm_revin_pred_data = np.flipud(convlstm_revin_preds_data[i])
        
        current_timestamp = timestamps[i]

        # Set overall title for the frame (date at the top)
        fig.suptitle(f'Date: {current_timestamp.strftime("%Y-%m-%d")}', fontsize=16, y=0.98) 

        # --- Row 1: Ground Truth (Left) | Ground Truth (Right) ---
        axes[0, 0].imshow(current_gt, cmap='viridis', aspect='auto', vmin=vmin_global, vmax=vmax_global)
        axes[0, 0].set_title('Ground Truth', fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(current_gt, cmap='viridis', aspect='auto', vmin=vmin_global, vmax=vmax_global)
        axes[0, 1].set_title('Ground Truth', fontsize=12)
        axes[0, 1].axis('off')

        # --- Row 2: ConvLSTM Prediction (Left) | ConvLSTM + RevIN Prediction (Right) ---
        axes[1, 0].imshow(current_convlstm_std_pred, cmap='viridis', aspect='auto', vmin=vmin_global, vmax=vmax_global)
        axes[1, 0].set_title('ConvLSTM Prediction', fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(current_convlstm_revin_pred_data, cmap='viridis', aspect='auto', vmin=vmin_global, vmax=vmax_global)
        axes[1, 1].set_title('ConvLSTM + RevIN Prediction', fontsize=12)
        axes[1, 1].axis('off')

        # --- Row 3: CNN-LSTM Prediction (Left) | CNN-LSTM + RevIN Prediction (Right) ---
        axes[2, 0].imshow(current_cnnlstm_std_pred, cmap='viridis', aspect='auto', vmin=vmin_global, vmax=vmax_global)
        axes[2, 0].set_title('CNN-LSTM Prediction', fontsize=12)
        axes[2, 0].axis('off')

        axes[2, 1].imshow(current_cnnlstm_revin_pred, cmap='viridis', aspect='auto', vmin=vmin_global, vmax=vmax_global)
        axes[2, 1].set_title('CNN-LSTM + RevIN Prediction', fontsize=12)
        axes[2, 1].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        frame_filename = os.path.join(FRAMES_DIR, f'frame_{i:04d}.png')
        plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"Generated frame {i+1}/{num_frames}")

    print(f"Finished generating {num_frames} frames in '{FRAMES_DIR}'.")
    return num_frames

def create_gif(num_frames):
    """Stitches the generated frames into a GIF."""
    print(f"\nCreating GIF: {GIF_OUTPUT_FILENAME}...")
    filenames = [os.path.join(FRAMES_DIR, f'frame_{i:04d}.png') for i in range(num_frames)]
    
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    
    imageio.mimsave(GIF_OUTPUT_FILENAME, images, fps=FPS)
    print(f"GIF saved as '{GIF_OUTPUT_FILENAME}'.")

    # Clean up individual frames (optional, uncomment if you want to remove them)
    # for filename in filenames:
    #     os.remove(filename)
    # os.rmdir(FRAMES_DIR)
    # print(f"Cleaned up '{FRAMES_DIR}' directory.")

# --- Main Execution ---
if __name__ == "__main__":
    # Load data for all four models
    cnnlstm_std_preds, cnnlstm_std_targets, cnnlstm_std_maes, cnnlstm_std_timestamps = load_model_data('cnnlstm')
    convlstm_std_preds, convlstm_std_targets, convlstm_std_maes, convlstm_std_timestamps = load_model_data('convlstm')
    cnnlstm_revin_preds, cnnlstm_revin_targets, cnnlstm_revin_maes, cnnlstm_revin_timestamps = load_model_data('cnnlstm_revin')
    convlstm_revin_preds_data, convlstm_revin_targets_data, convlstm_revin_maes_data, convlstm_revin_timestamps_data = load_model_data('convlstm_revin') # Renamed for clarity in this script

    # Check if all needed data is loaded
    if any(p is None for p in [cnnlstm_std_preds, convlstm_std_preds, cnnlstm_revin_preds, convlstm_revin_preds_data]):
        print("Cannot proceed with GIF generation due to missing data from one or more models.")
    else:
        # Use a common ground truth and timestamps (they should all be identical)
        ground_truth_data = cnnlstm_std_targets 
        common_timestamps = cnnlstm_std_timestamps

        # Create frames
        num_frames = create_gif_frames(ground_truth_data, cnnlstm_std_preds, convlstm_std_preds, cnnlstm_revin_preds, convlstm_revin_preds_data, common_timestamps)
        
        # Create GIF
        if num_frames > 0:
            create_gif(num_frames)
        else:
            print("No frames generated, GIF not created.")
