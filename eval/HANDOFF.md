# Handoff: fast sampling on the `speedy` branch

*Status as of 2026-11-26. Summary: the speed work is done and validated, but sample quality has not been checked. The existing quality plots are invalid, and fixing them is the first thing to do.*

## 1. Goal

Cut CPU inference time of the climemu emulator from ~5 s to under 1 s per call, **without retraining**. The only lever is the reverse-diffusion sampler: fewer or cheaper network evaluations.

## 2. What was done (June 15–20)

All of this is on the `speedy` branch. The code changes are in commit `128351d`, and the eval pipeline is in commit `3575bf2`.

### New sampler options

- **`DPMSolverVE`** ([src/diffusion/samplers/dpm_solver.py](../src/diffusion/samplers/dpm_solver.py)): DPM-Solver adapted to the variance-exploding schedule. It supports first-order and second-order (`solver="dpm2"`) updates.
- **Warm start / truncated diffusion (`t_start`)**: starts the reverse process at `t_start < 1` instead of at maximum noise, which skips the noisiest part of the trajectory. Only implemented for `DPMSolverVE`.
- **Euler parameterization** of the existing ODE sampler ([continuous_ode_sampler.py](../src/diffusion/samplers/continuous_ode_sampler.py)).
- These are exposed through `compile()`, and all defaults keep the old behavior:

  ```python
  from diffusion import DPMSolverVE
  emulator.compile(n_samples=1, n_steps=2, sampler_cls=DPMSolverVE, t_start=0.5)
  ```

  The signature is `compile(n_samples, n_steps=30, solver=None, sampler_cls=None, t_start=None)`. `.claude/decisions.md` also mentions a `dtype` option; it is **not** in the current signature.

### Speed results

MPI-ESM1-2-LR, `n_samples=1`, CPU. According to the branch notes, these runs were on an Apple Silicon laptop. The scripts are in [scripts/](../scripts/) (`bench*.py`).

| Configuration | Generation time | Speedup |
|---|---|---|
| Heun, 30 steps (reference) | 136.2 s | — |
| Heun, 2 steps (baseline) | 4.66 s | 1× |
| Euler parameterization | 2.4 s | 2× |
| `DPMSolverVE`, 2 steps | 0.29 s | 16× |
| **`DPMSolverVE`, 2 steps, `t_start=0.5`** | **0.147 s** | **32×** |

Dead ends: **bfloat16** gave no speedup on Apple Silicon CPU, and **XLA CPU flags** gave no measurable change.

### Quality check done so far (weak)

The best configuration (DPM, 2 steps, `t_start=0.5`) has RMSE = 3.96 against a 30-step Heun reference. Treat this number with caution. As implemented in [scripts/bench_truncated.py](../scripts/bench_truncated.py) and [scripts/gen_reference.py](../scripts/gen_reference.py), it is the pointwise RMSE of a **single sample**:

- `tas` only,
- one condition (`gmst=2.0`, `month=6`),
- one seed (42).

So it is not a distributional metric. With warm start, the two samples do not even follow the same noise trajectory. No acceptability threshold was ever set.

## 3. piControl evaluation pipeline

[eval/picontrol/](picontrol/) reruns the paper's MPI piControl diagnostics with a chosen sampler configuration:

| Script | Role |
|---|---|
| `inference.py` | Generates emulated piControl with `--sampler {heun,dpm} --n-steps N [--t-start T] --n-year Y --output-dir DIR`. Needs a GPU node. |
| `run_plot.py` | Runs a `paper/mpi/plots/piControl/<script>` against a given data folder and plot folder. |
| `plot_samples.py` | Plots a grid of random samples. |
| `run_all.sh` | Driver. Currently set to DPM 50 and DPM 50 warm with `N_YEAR=100`. The Heun 50 generation step is commented out. |

Generated data is in `/orcd/data/raffaele/001/shahineb/emulated/climemu/paper/mpi/outputs_<config>/emulated_piControl.nc`. On the cluster, `~shahineb/data` points to `/orcd/data/raffaele/001/shahineb`.

| Config | File size | Approx. length | Date |
|---|---|---|---|
| `dpm30`, `dpm30_warm` | 3.5 GB | 1000 years (default) | Jun 18 |
| `heun50`, `dpm50`, `dpm50_warm` | 354 MB | 100 years | Jun 18–19 |
| `outputs` (default, used by the paper) | 3.5 GB | 1000 years | **overwritten Jul 14** |
| `outputs_diffusion` | 3.5 GB | 1000 years | Jul 14. Owned by user `angela5`; origin unknown. |

There is no `heun30` data folder, although `plots/heun30/` exists.

Plots are in `eval/picontrol/plots/<config>/`. They are **git-ignored** (`*.jpg`), so they exist only in this working copy on the cluster.

## 4. Main finding: the existing eval plots are invalid

All plots in `eval/picontrol/plots/` are **byte-identical across all six configurations**: `heun30`, `heun50`, `dpm30`, `dpm30_warm`, `dpm50` and `dpm50_warm`. For example, every `emd.jpg` has md5 `a9e99ffe…` and every `moments.jpg` has md5 `cd42a534…`. This holds even for datasets of different lengths, so the plots cannot reflect the per-configuration data.

**Cause (found by reading the code; not yet confirmed by a rerun).** [run_plot.py](picontrol/run_plot.py) tries to redirect the data path like this:

```python
cfg.SamplingConfig.__dataclass_fields__['output_dir'].default = data_dir
```

A dataclass's `__init__` is generated when the class is defined, with the default values already built in, so this line has no effect. The plot modules call `Config()`, which builds `SamplingConfig()` through `default_factory`, and then read `config.sampling.output_dir`. That is always the default `.../paper/mpi/outputs`.

**Consequence.** Every plot shows whatever was in the default `outputs` folder on Jun 18–19, and that folder has since been overwritten (Jul 14). **No conclusion about DPM or warm-start quality can be drawn from the current plots.**

**Suggested fix** (not applied). In `run_plot.py`, patch the `Config` constructor before importing the plot module. The plot modules do `from paper.mpi.config import Config` at import time, so they will pick up the patched version:

```python
import paper.mpi.config as cfg
_Config = cfg.Config
def _patched_config(*args, **kwargs):
    c = _Config(*args, **kwargs)
    c.sampling.output_dir = data_dir
    return c
cfg.Config = _patched_config
```

How to check the fix: rerun two configurations and confirm that the plot md5 hashes now differ. Also print the path that `load_data` opens.

## 5. Open problems, roughly in priority order

1. **Fix `run_plot.py` and regenerate all plots** from the existing data. No new inference is needed. The plot scripts load 3.5 GB files into memory, so run them as a batch job, not on the login node.
2. **Make comparisons like-for-like.** The 30-step configurations have 1000 years of data and the 50-step ones have 100 years. Compare each fast configuration against a Heun reference of the **same length**. Heun 30 / 1000-year data must be regenerated, because the default `outputs` folder was overwritten. The fast configurations that matter in practice (2–3 steps, with and without `t_start`) have **not been run through the eval at all**.
3. **Define an acceptance criterion.** For example: EMD-to-noise ratio, moments and PSD within X of Heun 30, for each variable and season. Without one, "fast enough and good enough" can't be decided.
4. **Extend to the other ESMs**: MIROC6, ACCESS-ESM1-5, CanESM5 and IPSL-CM6A-LR. Everything so far is MPI-ESM1-2-LR only. `paper/access/` has the equivalent configuration and inference code for ACCESS.
5. **Decide whether `DPMSolverVE` (with some `t_start`) becomes the default** in `compile()`. Currently the default is Heun with 30 steps.
6. **Add unit tests for `DPMSolverVE`** (none yet) and the `t_start` path. Rerun the test suite; it last passed (31 tests) on Jun 18.
7. **Document the new `compile()` options** in the README and docstrings. Either remove the stale `dtype` mention from `.claude/decisions.md` or implement the option.

## 6. Other notes

- Branch planning and decision notes: `.claude/plan.md` and `.claude/decisions.md`. Note that `plan.md` marks the work as "Complete" and does not reflect the eval findings above.
- The original paper diagnostics are in `paper/mpi/plots/piControl/`. The eval pipeline reuses them unchanged.
