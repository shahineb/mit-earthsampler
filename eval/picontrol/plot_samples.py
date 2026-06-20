"""Plot a 4x4 grid of random emulated piControl samples.

Rows are variables (Temperature, Precipitation, Relative Humidity, Wind Speed),
columns are 4 randomly selected (year, month) samples. PlateCarree projection
with divergent colormaps and tight spacing.

Usage:
    python eval/picontrol/plot_samples.py <data_dir> <output_dir> [--seed 42]

Example:
    python eval/picontrol/plot_samples.py \
        /home/shahineb/data/emulated/climemu/paper/mpi/outputs_dpm30 \
        eval/picontrol/plots/dpm30
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_src_root = os.path.join(_repo_root, 'src')
for p in [_repo_root, _src_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from paper.mpi.config import Config  # noqa: E402
from paper.mpi.plots.piControl.utils import load_data, VARIABLES, save_plot  # noqa: E402

VARIABLES_ORDER = ['tas', 'pr', 'hurs', 'sfcWind']
N_SAMPLES = 4
DPI = 300


def parse_args():
    parser = argparse.ArgumentParser(description="Plot random sample grids")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing emulated_piControl.nc")
    parser.add_argument("output_dir", type=str,
                        help="Directory to save plot")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Patch config to point to custom data
    import paper.mpi.config as cfg
    cfg.SamplingConfig.__dataclass_fields__['output_dir'].default = args.data_dir

    config = Config()
    _, piControl_diffusion, _ = load_data(config, in_memory=True)

    # Pick random (year, month) pairs
    rng = np.random.default_rng(args.seed)
    n_years = piControl_diffusion.sizes['year']
    n_months = piControl_diffusion.sizes['month']
    year_idx = rng.integers(0, n_years, size=N_SAMPLES)
    month_idx = rng.integers(0, n_months, size=N_SAMPLES)

    # GridSpec: 4 variable rows + 4 sample columns + thin colorbar column
    n_vars = len(VARIABLES_ORDER)
    width_ratios = [1] * N_SAMPLES + [0.03]
    height_ratios = [1] * n_vars

    fig = plt.figure(figsize=(N_SAMPLES * 3.5, n_vars * 1.8))
    gs = gridspec.GridSpec(n_vars, N_SAMPLES + 1,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios,
                           wspace=0.02, hspace=0.04)

    for i, var in enumerate(VARIABLES_ORDER):
        var_info = VARIABLES[var]
        cmap = var_info['cmap']

        # Collect samples and compute symmetric color limits
        samples = []
        for j in range(N_SAMPLES):
            sample = piControl_diffusion[var].isel(
                year=int(year_idx[j]), month=int(month_idx[j]))
            samples.append(sample)

        all_vals = np.concatenate([s.values.ravel() for s in samples])
        vmax = np.nanquantile(np.abs(all_vals), 0.99)

        mesh = None
        for j, sample in enumerate(samples):
            ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
            mesh = sample.plot.pcolormesh(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap=cmap, add_colorbar=False,
                vmin=-vmax, vmax=vmax)
            ax.coastlines(linewidth=0.3, color='0.3')
            ax.set_global()
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')

            # Column headers on top row
            if i == 0:
                yr = int(piControl_diffusion.year.values[year_idx[j]])
                mo = int(piControl_diffusion.month.values[month_idx[j]])
                ax.set_title(f"yr {yr}, mo {mo}", fontsize=9, weight='bold',
                             pad=2)

        # Row label on left
        ax_left = fig.add_subplot(gs[i, 0])
        # We already have the cartopy axis at gs[i,0], so use text on figure
        # Get the position of the first axis in this row
        pos = fig.axes[i * (N_SAMPLES + 0)].get_position()
        fig.text(pos.x0 - 0.02, (pos.y0 + pos.y1) / 2,
                 var_info['name'], va='center', ha='right',
                 rotation='vertical', fontsize=11, weight='bold')

        # Colorbar per row
        cax = fig.add_subplot(gs[i, -1])
        cbar = fig.colorbar(mesh, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(f"[{var_info['unit']}]", fontsize=8)

    os.makedirs(args.output_dir, exist_ok=True)
    save_plot(fig, args.output_dir, 'random_samples.jpg', dpi=DPI)
    print(f"Saved to {args.output_dir}/random_samples.jpg")


if __name__ == '__main__':
    main()
