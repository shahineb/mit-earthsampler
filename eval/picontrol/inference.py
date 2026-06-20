"""Generate emulated piControl samples with configurable sampler.

Supports DPM-Solver and warm-start (truncated diffusion) in addition to the
default Heun ODE sampler. Reuses data loading and saving from the paper's
piControl inference script.

Usage:
    # DPM-Solver, 30 steps
    python eval/picontrol/inference.py --sampler dpm --n-steps 30 --output-dir /path/to/output

    # DPM-Solver + warm-start
    python eval/picontrol/inference.py --sampler dpm --n-steps 30 --t-start 0.5 --output-dir /path/to/output

    # Default Heun sampler
    python eval/picontrol/inference.py --sampler heun --n-steps 30 --output-dir /path/to/output
"""
import os
import sys
import math
import argparse

# Ensure repo root and src/ are on path so local packages shadow site-packages
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_src_root = os.path.join(_repo_root, 'src')
for p in [_repo_root, _src_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import equinox as eqx  # noqa: E402
from tqdm import tqdm  # noqa: E402

from diffusion import ContinuousVESchedule, DPMSolverVE  # noqa: E402
from climemu.emulators.bouabid2025 import create_sampler, denormalize  # noqa: E402

from paper.mpi.config import Config  # noqa: E402
from paper.mpi.data import load_dataset  # noqa: E402
from paper.mpi.inference.piControl import (  # noqa: E402
    load_model_and_data, make_dataloader, save_predictions
)


def make_draw_samples_batch(nn, schedule, n_samples, n_steps, μ, σ,
                            output_size, sampler_cls=None, t_start=None):
    """Create a jitted+vmapped batch sampling function.

    We define draw_single as a closure so sampler_cls (a Python class) and
    t_start are captured as constants rather than passed through vmap.
    """
    warm_start_kwargs = {}
    if t_start is not None and sampler_cls is DPMSolverVE:
        mean_flat = jnp.zeros(math.prod(output_size))
        warm_start_kwargs["warm_start"] = (mean_flat, t_start)

    def draw_single(pattern, key):
        sampler = create_sampler(nn, schedule, pattern, μ, σ, output_size,
                                 sampler_cls=sampler_cls)
        samples = sampler.sample(n_samples, steps=n_steps, key=key,
                                 **warm_start_kwargs)
        return denormalize(samples, μ[:-1], σ[:-1])

    @eqx.filter_jit
    def draw_batch(pattern_batch, key):
        keys = jr.split(key, pattern_batch.shape[0])
        return jax.vmap(draw_single)(pattern_batch, keys)

    return draw_batch


def parse_args():
    parser = argparse.ArgumentParser(description="piControl inference with configurable sampler")
    parser.add_argument("--sampler", choices=["heun", "dpm"], default="heun",
                        help="Sampler type (default: heun)")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Number of diffusion steps (default: 30)")
    parser.add_argument("--t-start", type=float, default=None,
                        help="Warm-start t_start for truncated diffusion (DPM only)")
    parser.add_argument("--n-year", type=int, default=1000,
                        help="Number of years to generate (default: 1000)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save emulated_piControl.nc")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve sampler config
    sampler_cls = DPMSolverVE if args.sampler == "dpm" else None
    t_start = args.t_start if args.sampler == "dpm" else None

    config_label = args.sampler
    if t_start is not None:
        config_label += f"_t{t_start}"
    config_label += f"_{args.n_steps}steps"
    print(f"Config: {config_label}")
    print(f"  sampler_cls = {sampler_cls}")
    print(f"  n_steps     = {args.n_steps}")
    print(f"  t_start     = {t_start}")
    print(f"  output_dir  = {args.output_dir}")

    # Load model and data
    config = Config()
    model, β, lat, lon, μ_train, σ_train, σmax = load_model_and_data(config)

    # Estimate piControl GMST standard deviation
    piControl = load_dataset(root=config.data.root_dir,
                             model=config.data.model_name,
                             experiments=["piControl"],
                             variables=["tas"],
                             in_memory=False,
                             external_β=β)
    σpiControl = piControl.gmst['piControl']['tas'].std().item()
    print(f"σpiControl = {σpiControl}")

    # Prepare data loader
    key = jr.PRNGKey(0)
    nlat, nlon = config.model.input_size[1], config.model.input_size[2]
    test_loader = make_dataloader(args.n_year, β, σpiControl, key, nlat, nlon)

    # Initialize noise schedule
    schedule = ContinuousVESchedule(config.schedule.sigma_min, σmax)
    χtest = jr.PRNGKey(config.sampling.random_seed)

    # Initialize sampling function
    output_size = (config.model.out_channels, nlat, nlon)
    generate_samples = make_draw_samples_batch(
        nn=model, schedule=schedule,
        n_samples=1, n_steps=args.n_steps,
        μ=μ_train, σ=σ_train,
        output_size=output_size,
        sampler_cls=sampler_cls,
        t_start=t_start)

    # Generate predictions
    pred_samples = []
    with tqdm(total=args.n_year) as pbar:
        for batch_idx, pattern_batch in enumerate(test_loader):
            batch_pred_samples = generate_samples(pattern_batch, key=χtest)
            pred_samples.append(jax.device_get(batch_pred_samples))
            χtest, _ = jr.split(χtest)

            if (batch_idx + 1) % 100 == 0:
                save_predictions(pred_samples, lat, lon,
                                 args.output_dir, config.data.variables)
            _ = pbar.update(1)

    # Save final version
    save_predictions(pred_samples, lat, lon, args.output_dir, config.data.variables)
    print("Done.")


if __name__ == "__main__":
    main()
