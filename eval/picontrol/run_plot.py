"""Run a piControl plot script with custom data and output directories.

Monkey-patches Config.SamplingConfig to point to a custom emulated data dir,
then imports the specified plot module (which triggers module-level data loading
with the patched config) and runs its main() function.

Must be run as a separate process per plot (module-level code only executes once).

Usage:
    python eval/picontrol/run_plot.py <data_dir> <plot_dir> <script_name>

Example:
    python eval/picontrol/run_plot.py /path/to/dpm30/outputs eval/picontrol/plots/dpm30 plot_emd
"""
import sys
import os
import importlib

# Ensure repo root and src/ are on path so local packages shadow site-packages
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_src_root = os.path.join(_repo_root, 'src')
for p in [_repo_root, _src_root]:
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <data_dir> <plot_dir> <script_name>")
        sys.exit(1)

    data_dir = sys.argv[1]
    plot_dir = sys.argv[2]
    script_name = sys.argv[3]

    # Validate data exists
    nc_path = os.path.join(data_dir, "emulated_piControl.nc")
    if not os.path.exists(nc_path):
        print(f"Error: {nc_path} not found")
        sys.exit(1)

    print(f"Running {script_name}")
    print(f"  data_dir = {data_dir}")
    print(f"  plot_dir = {plot_dir}")

    # Patch Config before plot module is imported
    import paper.mpi.config as cfg
    cfg.SamplingConfig.__dataclass_fields__['output_dir'].default = data_dir

    # Import plot module — triggers module-level data loading with patched config
    module = importlib.import_module(f"paper.mpi.plots.piControl.{script_name}")

    # Override plot output directory
    module.OUTPUT_DIR = plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    # Run
    module.main()
    print(f"  Done: {script_name}")


if __name__ == "__main__":
    main()
