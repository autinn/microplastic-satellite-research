import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import json
from datetime import datetime
import os
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches 

# --- Configuration ---
MODEL_PREFIX_TO_ANALYZE = 'convlstm' 

GIF_OUTPUT_FILENAME = f'{MODEL_PREFIX_TO_ANALYZE}_clustering_analysis.gif'
FRAMES_DIR = f'gif_frames_{MODEL_PREFIX_TO_ANALYZE}_clusters'
FPS = 5
MAX_FRAMES = 100 

# --- DBSCAN Parameters ---
CONCENTRATION_THRESHOLD = 0.75 
DBSCAN_EPS = 5 
DBSCAN_MIN_SAMPLES = 5 

# --- Helper Functions ---
def load_model_data(model_prefix):
    """Loads predictions, targets, and timestamps for a given model prefix."""
    try:
        predictions = np.load(f'{model_prefix}_predictions.npy')
        targets = np.load(f'{model_prefix}_targets.npy')
        
        with open(f'{model_prefix}_timestamps.json', 'r') as f:
            timestamps_str = json.load(f)
            timestamps = [datetime.strptime(ts, "%Y-%m-%d") for ts in timestamps_str]
        
        print(f"Successfully loaded data for {model_prefix.upper()}.")
        return predictions, targets, timestamps
    except FileNotFoundError as e:
        print(f"Error: Could not find data files for {model_prefix.upper()}. Please ensure you have run the corresponding model script first.")
        print(e)
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading data for {model_prefix.upper()}: {e}")
        return None, None, None

def get_clusters_and_bounding_boxes(image_2d, threshold, eps, min_samples):
    """
    Applies DBSCAN to identify clusters of high concentration and returns bounding boxes.
    """
    high_conc_indices = np.where(image_2d > threshold)
    
    if high_conc_indices[0].size == 0:
        return [] 

    X_dbscan = np.column_stack((high_conc_indices[0], high_conc_indices[1]))

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_dbscan)
    labels = db.labels_

    unique_labels = set(labels)
    bounding_boxes = []

    for k in unique_labels:
        if k == -1: 
            continue

        class_member_mask = (labels == k)
        cluster_points = X_dbscan[class_member_mask]

        if len(cluster_points) > 0:
            min_row, max_row = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
            min_col, max_col = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])
            
            cluster_values = image_2d[cluster_points[:, 0], cluster_points[:, 1]]
            avg_conc_in_cluster = np.mean(cluster_values)
            
            bounding_boxes.append({
                'label': k,
                'bbox_coords': (min_col, min_row, max_col - min_col, max_row - min_row), 
                'avg_conc': avg_conc_in_cluster
            })
    return bounding_boxes

def create_gif_frames_single_model(ground_truth_data, model_pred_data, timestamps, model_name):
    """
    Creates individual frames (PNG images) for the GIF animation for a single model.
    Layout: 1x2 (Ground Truth | Model Prediction with clusters)
    """
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)

    num_frames = min(MAX_FRAMES, len(ground_truth_data))
    print(f"\nGenerating {num_frames} GIF frames for {model_name}...")

    vmin_global = 0.0
    vmax_global = 1.0 

    img_height, img_width = ground_truth_data.shape[1], ground_truth_data.shape[2]
    
    # --- FIX: Calculate fixed figure dimensions with more padding ---
    target_dpi = 100 
    
    # Define a base height for the image display area
    base_image_height_inches = 6.0 
    image_aspect_ratio = img_width / img_height
    base_image_width_inches = base_image_height_inches * image_aspect_ratio

    # Total figure size (including space for titles, and now more padding)
    # Increased padding by adjusting the constant values
    padding_factor_width = 2.0 # Increased from 1.5
    padding_factor_height = 2.0 # Increased from 1.5

    fig_width_inches = (base_image_width_inches * 2) + padding_factor_width 
    fig_height_inches = base_image_height_inches + padding_factor_height
    
    for i in range(num_frames):
        # Create figure with fixed size
        fig, axes = plt.subplots(1, 2, figsize=(fig_width_inches, fig_height_inches), dpi=target_dpi)
        
        current_gt = np.flipud(ground_truth_data[i])
        current_pred = np.flipud(model_pred_data[i])
        
        current_timestamp = timestamps[i]

        fig.suptitle(f'Date: {current_timestamp.strftime("%Y-%m-%d")}\n{model_name} Hotspot Analysis', fontsize=16, y=0.98) 

        # --- Plot Ground Truth with Clusters ---
        gt_bboxes = get_clusters_and_bounding_boxes(ground_truth_data[i], CONCENTRATION_THRESHOLD, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
        axes[0].imshow(current_gt, cmap='viridis', aspect='auto', vmin=vmin_global, vmax=vmax_global)
        axes[0].set_title('Ground Truth', fontsize=12)
        axes[0].axis('off')
        for bbox_info in gt_bboxes:
            rect = patches.Rectangle((bbox_info['bbox_coords'][0], current_gt.shape[0] - bbox_info['bbox_coords'][1] - bbox_info['bbox_coords'][3]), 
                                     bbox_info['bbox_coords'][2], bbox_info['bbox_coords'][3], 
                                     linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text(bbox_info['bbox_coords'][0] + bbox_info['bbox_coords'][2] / 2, 
                         current_gt.shape[0] - bbox_info['bbox_coords'][1] - bbox_info['bbox_coords'][3] - 5, 
                         f'{bbox_info["avg_conc"]:.2f}', 
                         color='red', fontsize=8, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))


        # --- Plot Model Prediction with Clusters ---
        pred_bboxes = get_clusters_and_bounding_boxes(model_pred_data[i], CONCENTRATION_THRESHOLD, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
        axes[1].imshow(current_pred, cmap='viridis', aspect='auto', vmin=vmin_global, vmax=vmax_global)
        axes[1].set_title(f'{model_name} Prediction', fontsize=12)
        axes[1].axis('off')
        for bbox_info in pred_bboxes:
            rect = patches.Rectangle((bbox_info['bbox_coords'][0], current_pred.shape[0] - bbox_info['bbox_coords'][1] - bbox_info['bbox_coords'][3]), 
                                     bbox_info['bbox_coords'][2], bbox_info['bbox_coords'][3], 
                                     linewidth=2, edgecolor='red', facecolor='none')
            axes[1].add_patch(rect)
            axes[1].text(bbox_info['bbox_coords'][0] + bbox_info['bbox_coords'][2] / 2, 
                         current_pred.shape[0] - bbox_info['bbox_coords'][1] - bbox_info['bbox_coords'][3] - 5, 
                         f'{bbox_info["avg_conc"]:.2f}', 
                         color='red', fontsize=8, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # --- FIX: Use bbox_inches='tight' with pad_inches for more control ---
        # Increased pad_inches for more whitespace around the plot content
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        frame_filename = os.path.join(FRAMES_DIR, f'frame_{i:04d}.png')
        # Increased pad_inches to add more whitespace around the content
        plt.savefig(frame_filename, dpi=target_dpi, bbox_inches='tight', pad_inches=0.5) # Increased pad_inches
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
    # Load data for the selected model
    model_preds, model_targets, model_timestamps = load_model_data(MODEL_PREFIX_TO_ANALYZE)

    if model_preds is None:
        print(f"Cannot proceed with GIF generation. Data for {MODEL_PREFIX_TO_ANALYZE} not loaded.")
    else:
        # Create frames
        num_frames = create_gif_frames_single_model(model_targets, model_preds, model_timestamps, MODEL_PREFIX_TO_ANALYZE.replace('_', ' ').upper())
        
        # Create GIF
        if num_frames > 0:
            create_gif(num_frames)
        else:
            print("No frames generated, GIF not created.")
