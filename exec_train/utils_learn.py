import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

# Brighten colors
def brighten_colors(cmap, factor=0.2):
    c = cmap(np.arange(cmap.N))
    c[:, :-1] += (1.0 - c[:, :-1]) * factor
    c = np.clip(c, 0, 1)
    return ListedColormap(c)

# Get colormap
def get_fold_colormap(fold, brighten=True):
    base_colormaps = [cm.plasma, cm.viridis, cm.inferno, cm.magma, cm.cividis]
    if brighten:
        return brighten_colors(base_colormaps[fold % len(base_colormaps)])
    else:
        return base_colormaps[fold % len(base_colormaps)]

# Plot Diagnostics
def plot_learn(loss_data, output_dir):
    num_losses = len(loss_data['train'])
    fig, axes = plt.subplots(num_losses, 1, figsize=(10, 5 * num_losses))
    if num_losses == 1:
        axes = [axes]

    for idx, loss_name in enumerate(loss_data['train'].keys()):
        train_losses = np.array([float(x) for x in loss_data['train'][loss_name]])
        val_losses = np.array([float(x) for x in loss_data['val'][loss_name]])

        # Prepare data for gradient plot
        x_train = np.arange(len(train_losses))
        y_train = np.array(train_losses)
        points_train = np.array([x_train, y_train]).T.reshape(-1, 1, 2)
        segments_train = np.concatenate([points_train[:-1], points_train[1:]], axis=1)

        x_val = np.arange(len(val_losses))
        y_val = np.array(val_losses)
        points_val = np.array([x_val, y_val]).T.reshape(-1, 1, 2)
        segments_val = np.concatenate([points_val[:-1], points_val[1:]], axis=1)

        # Set up the plot for training and validation loss
        ax = axes[idx]
        ax.set_title(f'{loss_name.replace("_", " ").title()} (Train vs. Val)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)

        # Training loss plot
        colormap_train = get_fold_colormap(0, brighten=True)
        norm_train = plt.Normalize(y_train.min(), y_train.max())
        lc_train = LineCollection(segments_train, cmap=colormap_train, norm=norm_train)
        lc_train.set_array(y_train)
        lc_train.set_linewidth(2)
        ax.add_collection(lc_train)

        # Validation loss plot
        colormap_val = get_fold_colormap(3, brighten=True)
        norm_val = plt.Normalize(y_val.min(), y_val.max())
        lc_val = LineCollection(segments_val, cmap=colormap_val, norm=norm_val)
        lc_val.set_array(y_val)
        lc_val.set_linewidth(2)
        ax.add_collection(lc_val)

        # Axis and grid setup
        ax.grid(True, linestyle='--', linewidth=0.5)
        max_iterations = max(len(train_losses), len(val_losses))
        tick_interval = max(1, max_iterations // 15)
        ax.set_xticks(np.arange(0, max_iterations + 1, tick_interval))
        ax.set_yticks(np.linspace(min(y_train.min(), y_val.min()), max(y_train.max(), y_val.max()), num=15))
        ax.autoscale_view()
        for _, spine in ax.spines.items():
            spine.set_linewidth(2)

         # Legend setup
        patch_train = Patch(color=colormap_train(norm_train(y_train.min())), label=f'{loss_name.title()} Train')
        patch_val = Patch(color=colormap_val(norm_val(y_val.min())), label=f'{loss_name.title()} Val')
        ax.legend(handles=[patch_train, patch_val], loc='upper right', fontsize=10)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_curves.png')
    plt.show()