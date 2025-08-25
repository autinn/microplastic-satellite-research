import os
import numpy as np
import matplotlib.pyplot as plt

MODELS = [
    ('cnnlstm', 'cnnlstm_error_maps.npy'),
    ('convlstm', 'convlstm_error_maps.npy'),
    ('cnnlstm_revin', 'cnnlstm_revin_error_maps.npy'),
    ('convlstm_revin', 'convlstm_revin_error_maps.npy'),
]


def main():
    rows = 1
    cols = len(MODELS)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4))
    if cols == 1:
        axes = [axes]

    vmax_global = None
    mae_maps = []
    labels = []

    # Load and compute MAE maps
    for name, fname in MODELS:
        if not os.path.exists(fname):
            print(f"Missing {fname}, skipping {name}")
            continue
        errs = np.load(fname)  # (N, H, W)
        mae_map = errs.mean(axis=0)
        mae_maps.append(mae_map)
        labels.append(name)
        vmax = float(np.percentile(mae_map, 99.0))
        vmax_global = vmax if vmax_global is None else max(vmax_global, vmax)

    # Plot
    for idx, mae_map in enumerate(mae_maps):
        ax = axes[idx]
        im = ax.imshow(mae_map, cmap='magma', origin='lower', vmin=0.0, vmax=vmax_global)
        ax.set_title(f"MAE: {labels[idx]}")
        ax.axis('off')

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Absolute Error')
    plt.tight_layout()
    plt.savefig('mae_maps_all_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved mae_maps_all_models.png")


if __name__ == '__main__':
    main()
