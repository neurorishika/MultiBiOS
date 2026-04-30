#!/bin/bash
# Example: Run protocol with real-time progress monitoring
# This script demonstrates how to use the --progress flag

echo "=== MultiBiOS Protocol with Real-Time Progress ==="
echo ""

# Activate conda environment (adjust if needed)
echo "Activating multibios environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multibios

# Run protocol with progress monitoring enabled
echo ""
echo "Running protocol with real-time progress monitoring..."
echo "  - Progress updates every 100ms"
echo "  - Verbose logging enabled"
echo ""

python multibios/run_protocol.py \
    --yaml protocols/example_protocol.yaml \
    --hardware config/hardware.yaml \
    --verbose \
    --progress \
    --progress-interval 100 \
    --interactive

echo ""
echo "=== Protocol Complete ==="
echo "Check data/runs/ for output files and preview.html"
