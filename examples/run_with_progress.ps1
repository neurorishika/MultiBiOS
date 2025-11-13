# Example: Run protocol with real-time progress monitoring
# This script demonstrates how to use the --progress flag

Write-Host "=== MultiBiOS Protocol with Real-Time Progress ===" -ForegroundColor Cyan
Write-Host ""

# Activate conda environment (adjust if needed)
Write-Host "Activating multibios environment..." -ForegroundColor Yellow
conda activate multibios

# Run protocol with progress monitoring enabled
Write-Host ""
Write-Host "Running protocol with real-time progress monitoring..." -ForegroundColor Yellow
Write-Host "  - Progress updates every 100ms" -ForegroundColor Gray
Write-Host "  - Verbose logging enabled" -ForegroundColor Gray
Write-Host ""

python multibios/run_protocol.py `
    --yaml config/example_protocol.yaml `
    --hardware config/hardware.yaml `
    --verbose `
    --progress `
    --progress-interval 100 `
    --interactive

Write-Host ""
Write-Host "=== Protocol Complete ===" -ForegroundColor Green
Write-Host "Check data/runs/ for output files and preview.html" -ForegroundColor Gray
