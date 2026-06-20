#!/bin/bash
# Evaluate DPM-Solver quality on piControl.
#
# Generates emulated piControl samples with different sampler configs,
# then recreates all 8 diagnostic plots for each config.
#
# Prerequisites:
#   - GPU node (h100) with sandbox environment activated
#
# Usage:
#   bash eval/picontrol/run_all.sh
set -e

BASE_DATA="/home/shahineb/data/emulated/climemu/paper/mpi/outputs"
BASE_PLOT="eval/picontrol/plots"
N_YEAR=100

SCRIPTS=(
    plot_emd
    plot_moments
    plot_power_spectra
)

# ─── Config 1: Heun 50-step baseline ───────────────────
echo "=== Generating Heun 50-step baseline samples ==="
# python eval/picontrol/inference.py \
#     --sampler heun --n-steps 50 --n-year $N_YEAR \
#     --output-dir "${BASE_DATA}_heun50"

# echo "=== Plotting Heun 50-step baseline ==="
# for s in "${SCRIPTS[@]}"; do
#     python eval/picontrol/run_plot.py "${BASE_DATA}_heun50" "$BASE_PLOT/heun50" "$s"
# done

# ─── Config 2: DPM-Solver 50 steps ───────────────────────────────────────────
echo "=== Generating DPM-Solver 50-step samples ==="
python eval/picontrol/inference.py \
    --sampler dpm --n-steps 50 --n-year $N_YEAR \
    --output-dir "${BASE_DATA}_dpm50"

echo "=== Plotting DPM-Solver 50-step ==="
for s in "${SCRIPTS[@]}"; do
    python eval/picontrol/run_plot.py "${BASE_DATA}_dpm50" "$BASE_PLOT/dpm50" "$s"
done

# ─── Config 3: DPM-Solver 50 steps + warm-start (t_start=0.5) ────────────────
echo "=== Generating DPM-Solver 50-step + warm-start samples ==="
python eval/picontrol/inference.py \
    --sampler dpm --n-steps 50 --t-start 0.5 --n-year $N_YEAR \
    --output-dir "${BASE_DATA}_dpm50_warm"

echo "=== Plotting DPM-Solver 50-step + warm-start ==="
for s in "${SCRIPTS[@]}"; do
    python eval/picontrol/run_plot.py "${BASE_DATA}_dpm50_warm" "$BASE_PLOT/dpm50_warm" "$s"
done

# ─── Random sample grids ─────────────────────────────────────────────────────
echo "=== Generating random sample grids ==="
python eval/picontrol/plot_samples.py "${BASE_DATA}_heun50"     "$BASE_PLOT/heun50"
python eval/picontrol/plot_samples.py "${BASE_DATA}_dpm50"      "$BASE_PLOT/dpm50"
python eval/picontrol/plot_samples.py "${BASE_DATA}_dpm50_warm" "$BASE_PLOT/dpm50_warm"

echo "=== All done ==="
echo "Plots saved to:"
echo "  $BASE_PLOT/heun50/"
echo "  $BASE_PLOT/dpm50/"
echo "  $BASE_PLOT/dpm50_warm/"
