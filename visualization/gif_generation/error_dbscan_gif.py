import os
import json
import argparse
from typing import Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
from utils.repro_eval import evaluate_arrays

# --- Defaults ---
DEFAULT_MODEL_PREFIX = 'convlstm'  # e.g., cnnlstm | convlstm | cnnlstm_revin | convlstm_revin
DEFAULT_FPS = 2
DEFAULT_MAX_FRAMES = 200
DEFAULT_PRED_CMAP = 'viridis'
DEFAULT_ERR_CMAP = 'magma'
DEFAULT_ROBUST_PCT = 99.0  # percentile for error vmax
DEFAULT_THRESHOLD = 0.7
DEFAULT_DBSCAN_EPS = 4
DEFAULT_DBSCAN_MIN_SAMPLES = 4


# --- Timestamp helpers (compatible with existing files) ---
def derive_timestamp_paths(prefix: str) -> List[str]:
    candidates = [f'{prefix}_timestamps.json']
    if prefix == 'cnnlstm':
        candidates.append('lstm_timestamps.json')
    if prefix == 'convlstm':
        candidates.append('convlstm_timestamps.json')
    if prefix == 'cnnlstm_revin':
        candidates.append('cnnlstm_revin_timestamps.json')
    if prefix == 'convlstm_revin':
        candidates.append('convlstm_revin_timestamps.json')
    return candidates


def load_timestamps_for_prefix(prefix: str) -> Optional[List[str]]:
    for p in derive_timestamp_paths(prefix):
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def format_model_name(prefix: str) -> str:
    """Format a model prefix for display in titles."""
    return prefix.replace('_', ' ').upper()


# --- IO helpers ---
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_error_maps(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    return np.abs(pred - target)


# --- DBSCAN helpers ---
def get_clusters_and_bounding_boxes(image_2d: np.ndarray, threshold: float, eps: float, min_samples: int):
    """
    Applies DBSCAN to identify clusters of high concentration and returns bounding boxes.
    Returns a list of dicts with keys: 'label', 'bbox_coords' (x, y, w, h), 'avg_conc'.
    Coordinates (x, y) are in array-space (col, row) prior to any display flipping.
    """
    high_conc_indices = np.where(image_2d > threshold)
    if high_conc_indices[0].size == 0:
        return []

    X = np.column_stack((high_conc_indices[0], high_conc_indices[1]))  # (row, col)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    bounding_boxes = []
    for k in set(labels):
        if k == -1:
            continue
        cluster_points = X[labels == k]
        if len(cluster_points) == 0:
            continue
        min_row, max_row = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
        min_col, max_col = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])
        avg_conc = float(np.mean(image_2d[cluster_points[:, 0], cluster_points[:, 1]]))
        bounding_boxes.append({
            'label': int(k),
            'bbox_coords': (int(min_col), int(min_row), int(max_col - min_col), int(max_row - min_row)),
            'avg_conc': avg_conc,
        })
    return bounding_boxes


# --- Frame builder ---
def build_frames(
    pred: np.ndarray,
    target: np.ndarray,
    err_maps: np.ndarray,
    timestamps: Optional[List[str]],
    model_name: str,
    frame_dir: str,
    pred_cmap: str,
    err_cmap: str,
    robust_pct: float,
    threshold: float,
    eps: float,
    min_samples: int,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    err_vmax: Optional[float] = None,
    max_frames: Optional[int] = None,
    metrics_text: Optional[str] = None,
):
    ensure_dir(frame_dir)

    # Shared limits
    if vmin is None:
        vmin = float(min(target.min(), pred.min()))
    if vmax is None:
        vmax = float(max(target.max(), pred.max()))
    if err_vmax is None:
        err_vmax = float(np.percentile(err_maps, robust_pct))
    err_vmin = 0.0

    num_frames = pred.shape[0]
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    frame_paths = []
    for i in range(num_frames):
        # Use flipping to align array index space with display coordinates for overlays
        gt_disp = np.flipud(target[i])
        pred_disp = np.flipud(pred[i])
        err_disp = np.flipud(err_maps[i])

        H, W = gt_disp.shape

        # Precompute DBSCAN bboxes in array index space (no flip)
        gt_bboxes = get_clusters_and_bounding_boxes(target[i], threshold, eps, min_samples)
        pred_bboxes = get_clusters_and_bounding_boxes(pred[i], threshold, eps, min_samples)

        # GridSpec with dedicated colorbar axis
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[0, 2])
        cax = fig.add_subplot(gs[0, 3])

        # NEW: Combine model name, timestamp, and metrics into a single suptitle
        full_suptitle = f"{model_name}"
        if timestamps is not None and i < len(timestamps):
            full_suptitle += f" — {timestamps[i]}"
        if metrics_text:
            full_suptitle += f"\n{metrics_text}"

        fig.suptitle(full_suptitle, fontsize=11, y=0.98)
        
        # Ground truth panel
        ax_gt.imshow(gt_disp, cmap=pred_cmap, origin='upper', vmin=vmin, vmax=vmax)
        ax_gt.set_title('Ground Truth + DBSCAN', fontsize=10, y=1.0)
        ax_gt.axis('off')
        for b in gt_bboxes:
            x, y, w, h = b['bbox_coords']
            # Transform row->display y due to vertical flip
            y_disp = H - (y + h)
            rect = patches.Rectangle((x, y_disp), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax_gt.add_patch(rect)
            ax_gt.text(x + w / 2, max(0, y_disp - 2), f"{b['avg_conc']:.2f}",
                       color='red', fontsize=8, ha='center', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        # Prediction panel
        ax_pred.imshow(pred_disp, cmap=pred_cmap, origin='upper', vmin=vmin, vmax=vmax)
        ax_pred.set_title(f'Prediction + DBSCAN', fontsize=10, y=1.0)
        ax_pred.axis('off')
        for b in pred_bboxes:
            x, y, w, h = b['bbox_coords']
            y_disp = H - (y + h)
            rect = patches.Rectangle((x, y_disp), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax_pred.add_patch(rect)
            ax_pred.text(x + w / 2, max(0, y_disp - 2), f"{b['avg_conc']:.2f}",
                         color='red', fontsize=8, ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        # Error panel
        im_err = ax_err.imshow(err_disp, cmap=err_cmap, origin='upper', vmin=err_vmin, vmax=err_vmax)
        ax_err.set_title(f'Absolute Error', fontsize=10, y=1.0)
        ax_err.axis('off')

        # Colorbar
        cbar = fig.colorbar(im_err, cax=cax)
        cbar.set_label('Absolute Error', rotation=90)

        # Reserve space at top for suptitle + metrics subtitle
        # top=0.88 is a good value to make sure metrics don't overlap with subplots
        fig.subplots_adjust(left=0.04, right=0.96, top=0.8, bottom=0.08, wspace=0.25)

        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)
    return frame_paths


def build_gif(frame_paths: List[str], gif_path: str, fps: int):
    with imageio.get_writer(gif_path, mode='I', duration=1.0 / max(1, fps)) as writer:
        for p in frame_paths:
            writer.append_data(imageio.imread(p))
    print(f"Saved {gif_path}")


def main():
    parser = argparse.ArgumentParser(description='Build 1x3 GIF with DBSCAN overlays and error panel')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PREFIX,
                        help='Model prefix (e.g., cnnlstm, convlstm, cnnlstm_revin, convlstm_revin)')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS)
    parser.add_argument('--max_frames', type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument('--pred_cmap', type=str, default=DEFAULT_PRED_CMAP)
    parser.add_argument('--err_cmap', type=str, default=DEFAULT_ERR_CMAP)
    parser.add_argument('--robust_pct', type=float, default=DEFAULT_ROBUST_PCT)
    parser.add_argument('--vmin', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None)
    parser.add_argument('--err_vmax', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='DBSCAN threshold for high concentration mask')
    parser.add_argument('--eps', type=float, default=DEFAULT_DBSCAN_EPS,
                        help='DBSCAN eps parameter')
    parser.add_argument('--min_samples', type=int, default=DEFAULT_DBSCAN_MIN_SAMPLES,
                        help='DBSCAN min_samples parameter')
    args = parser.parse_args()

    prefix = args.model
    pred_path = f'{prefix}_predictions.npy'
    target_path = f'{prefix}_targets.npy'
    ts = load_timestamps_for_prefix(prefix)

    if not (os.path.exists(pred_path) and os.path.exists(target_path)):
        print(f"Missing inputs for {prefix}. Expected {pred_path} and {target_path}.")
        return

    pred = np.load(pred_path)
    target = np.load(target_path)
    # squeeze channel if present
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if target.ndim == 4 and target.shape[-1] == 1:
        target = target[..., 0]

    # Load or compute error maps
    err_maps_path = f'{prefix}_error_maps.npy'
    if os.path.exists(err_maps_path):
        err_maps = np.load(err_maps_path)
    else:
        err_maps = compute_error_maps(pred, target)
        np.save(err_maps_path, err_maps)

    # Compute overall metrics for display
    metrics = evaluate_arrays(target, pred, tol=0.10)
    metrics_text = (
        f"R2={metrics['r2']:.3f}  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  Acc@10%={metrics['acc_within_10pct']:.3f}"
    )
    print(f"Overall metrics for {prefix}: {metrics_text}")

    frames_dir = f'gif_frames_{prefix}_error_dbscan'
    frame_paths = build_frames(
        pred=pred,
        target=target,
        err_maps=err_maps,
        timestamps=ts,
        model_name=format_model_name(prefix),
        frame_dir=frames_dir,
        pred_cmap=args.pred_cmap,
        err_cmap=args.err_cmap,
        robust_pct=args.robust_pct,
        threshold=args.threshold,
        eps=args.eps,
        min_samples=args.min_samples,
        vmin=args.vmin,
        vmax=args.vmax,
        err_vmax=args.err_vmax,
        max_frames=args.max_frames,
        metrics_text=metrics_text,
    )
    gif_path = f'error_dbscan_{prefix}.gif'
    build_gif(frame_paths, gif_path, fps=args.fps)


if __name__ == '__main__':
    main()
