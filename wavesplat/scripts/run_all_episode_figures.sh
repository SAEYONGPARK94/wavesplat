#!/bin/bash
# Generate GT vs WaveSplat heatmap figures for all 100 episodes.
# Output: wavesplat/thesis/figures_episodes/episode_*.png

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Generating figures for all 100 episodes ==="
echo "Project dir: $PROJECT_DIR"
echo "Output dir:  thesis/figures_episodes/"
echo ""

python3 scripts/generate_all_episode_figures.py

echo ""
echo "=== Complete ==="
echo "Generated files:"
ls -1 thesis/figures_episodes/*.png 2>/dev/null | wc -l
echo "figures in thesis/figures_episodes/"
