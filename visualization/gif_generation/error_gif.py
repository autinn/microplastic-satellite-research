import os
import json
import argparse
from typing import Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# --- Config (overridable via CLI) ---
DEFAULT_MODEL_PREFIX = 'convlstm'  # e.g., cnnlstm | convlstm | cnnlstm_revin | convlstm_revin
DEFAULT_FPS = 3
DEFAULT_MAX_FRAMES = 200
DEFAULT_PRED_CMAP = 'viridis'
DEFAULT_ERR_CMAP = 'magma'
DEFAULT_ROBUST_PCT = 99.0  # percentile for error vmax


def derive_timestamp_paths(prefix: str) -> List[str]:
    # Try common patterns; include legacy name for cnnlstm
    candidates = [
        f'{prefix}_timestamps.json',
    ]
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_error_maps(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    return np.abs(pred - target)


def build_frames(
    pred: np.ndarray,
    target: np.ndarray,
    err_maps: np.ndarray,
    timestamps: Optional[List[str]],
    frame_dir: str,
    pred_cmap: str,
    err_cmap: str,
    robust_pct: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    err_vmax: Optional[float] = None,
    max_frames: Optional[int] = None,
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
        # Use a GridSpec with a dedicated colorbar axis to avoid overlap/misalignment
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[0, 2])
        cax = fig.add_subplot(gs[0, 3])

        # Ground truth
        im_gt = ax_gt.imshow(target[i], cmap=pred_cmap, origin='lower', vmin=vmin, vmax=vmax)
        title_left = 'Ground Truth'
        if timestamps is not None and i < len(timestamps):
            title_left += f"\n{timestamps[i]}"
        ax_gt.set_title(title_left, fontsize=10)
        ax_gt.axis('off')

        # Prediction
        im_pred = ax_pred.imshow(pred[i], cmap=pred_cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax_pred.set_title('Prediction', fontsize=10)
        ax_pred.axis('off')

        # Error
        im_err = ax_err.imshow(err_maps[i], cmap=err_cmap, origin='lower', vmin=err_vmin, vmax=err_vmax)
        ax_err.set_title('Absolute Error', fontsize=10)
        ax_err.axis('off')

        # Colorbar placed in its own axis on the right
        cbar = fig.colorbar(im_err, cax=cax)
        cbar.set_label('Absolute Error', rotation=90)

        # Adjust spacing without tight_layout (avoids warning and misplacement)
        fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.08, wspace=0.25)

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
    parser = argparse.ArgumentParser(description='Build 1x3 error GIF for a specific model prefix')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PREFIX, help='Model prefix (e.g., cnnlstm, convlstm, cnnlstm_revin, convlstm_revin)')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS)
    parser.add_argument('--max_frames', type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument('--pred_cmap', type=str, default=DEFAULT_PRED_CMAP)
    parser.add_argument('--err_cmap', type=str, default=DEFAULT_ERR_CMAP)
    parser.add_argument('--robust_pct', type=float, default=DEFAULT_ROBUST_PCT)
    parser.add_argument('--vmin', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None)
    parser.add_argument('--err_vmax', type=float, default=None)
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

    frames_dir = f'gif_frames_{prefix}_error'
    frame_paths = build_frames(
        pred=pred,
        target=target,
        err_maps=err_maps,
        timestamps=ts,
        frame_dir=frames_dir,
        pred_cmap=args.pred_cmap,
        err_cmap=args.err_cmap,
        robust_pct=args.robust_pct,
        vmin=args.vmin,
        vmax=args.vmax,
        err_vmax=args.err_vmax,
        max_frames=args.max_frames,
    )
    gif_path = f'error_{prefix}.gif'
    build_gif(frame_paths, gif_path, fps=args.fps)


if __name__ == '__main__':
    main()
