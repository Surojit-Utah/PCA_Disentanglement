# Reproduce Paper Results (PowerShell version)
# This script runs all evaluations for AVAE disentanglement analysis
# to reproduce Table 1 and Figures 1-2 from the ICASSP 2024 paper.

param(
    [string]$Dataset = "both",
    [int]$NumSeeds = 10,
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: .\reproduce_paper_results.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Dataset {dsprites|shapes3d|both}  Dataset to evaluate (default: both)"
    Write-Host "  -NumSeeds N                        Number of random seeds (default: 10)"
    Write-Host "  -Help                              Show this help message"
    Write-Host ""
    Write-Host "Example:"
    Write-Host "  .\reproduce_paper_results.ps1 -Dataset shapes3d -NumSeeds 10"
    exit 0
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AVAE Disentanglement Evaluation Pipeline" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check environment variables
if (-not $env:DSPRITES_DATA_DIR) {
    Write-Host "ERROR: DSPRITES_DATA_DIR environment variable not set" -ForegroundColor Red
    Write-Host "Please set: `$env:DSPRITES_DATA_DIR='C:\path\to\dsprites.npz'"
    exit 1
}

if (-not $env:SHAPES3D_DATA_DIR) {
    Write-Host "ERROR: SHAPES3D_DATA_DIR environment variable not set" -ForegroundColor Red
    Write-Host "Please set: `$env:SHAPES3D_DATA_DIR='C:\path\to\shapes3d.npz'"
    exit 1
}

Write-Host "✓ Environment variables set" -ForegroundColor Green
Write-Host "  DSPRITES_DATA_DIR: $env:DSPRITES_DATA_DIR"
Write-Host "  SHAPES3D_DATA_DIR: $env:SHAPES3D_DATA_DIR"
Write-Host ""

Write-Host "Configuration:"
Write-Host "  Dataset: $Dataset"
Write-Host "  Number of seeds: $NumSeeds"
Write-Host ""

# Function to run evaluation for a dataset
function Run-Evaluation {
    param(
        [int]$ConfigId,
        [string]$DatasetName
    )
    
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Evaluating $DatasetName" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    for ($seed = 1; $seed -le $NumSeeds; $seed++) {
        Write-Host "-------------------------------------------" -ForegroundColor Yellow
        Write-Host "Run $seed / $NumSeeds" -ForegroundColor Yellow
        Write-Host "-------------------------------------------" -ForegroundColor Yellow
        
        # FactorVAE Score
        Write-Host "Computing FactorVAE score..."
        python eval\compute_metrics.py --config_id $ConfigId --metric factor_pca_axis --seed $seed
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        
        # MIG Score
        Write-Host "Computing MIG score..."
        python eval\compute_metrics.py --config_id $ConfigId --metric mig_pca_axis --seed $seed
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        
        # MSE
        Write-Host "Computing MSE..."
        python eval\compute_mse.py --config_id $ConfigId --seed $seed
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        
        Write-Host "Run $seed completed ✓" -ForegroundColor Green
        Write-Host ""
    }
    
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "$DatasetName evaluation complete!" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Print summary statistics
    Write-Host "Results Summary for $DatasetName:" -ForegroundColor Cyan
    Write-Host "-------------------------------------------"
    
    if ($DatasetName -eq "DSprites") {
        $factorPath = "Output\DSprites\Factor_PCA\No_Whiten\factor_pca_axis_metric_stat.txt"
        $migPath = "Output\DSprites\MIG_PCA\No_Whiten\mig_pca_axis_metric_stat.txt"
        $msePath = "Output\DSprites\mse_error.txt"
        
        if (Test-Path $factorPath) {
            Write-Host "FactorVAE Score (Expected: 79.17 ± 1.64):"
            Get-Content $factorPath
        }
        if (Test-Path $migPath) {
            Write-Host "MIG Score (Expected: 0.20 ± 0.02):"
            Get-Content $migPath
        }
        if (Test-Path $msePath) {
            Write-Host "MSE (Expected: 2.98 ± 0.28):"
            Get-Content $msePath
        }
    } else {
        $factorPath = "Output\Shapes3D\Factor_PCA\No_Whiten\factor_pca_axis_metric_stat.txt"
        $migPath = "Output\Shapes3D\MIG_PCA\No_Whiten\mig_pca_axis_metric_stat.txt"
        $msePath = "Output\Shapes3D\mse_error.txt"
        
        if (Test-Path $factorPath) {
            Write-Host "FactorVAE Score (Expected: 91.93 ± 3.27):"
            Get-Content $factorPath
        }
        if (Test-Path $migPath) {
            Write-Host "MIG Score (Expected: 0.67 ± 0.04):"
            Get-Content $migPath
        }
        if (Test-Path $msePath) {
            Write-Host "MSE (Expected: 10.29 ± 0.37):"
            Get-Content $msePath
        }
    }
    
    Write-Host ""
}

# Generate visualizations
function Generate-Visualizations {
    param(
        [int]$ConfigId,
        [string]$DatasetName
    )
    
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Generating Visualizations for $DatasetName" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Angle plots (Figure 2)
    Write-Host "Generating angle distribution plots (Figure 2)..."
    python plot_angles.py --config_id $ConfigId --seed 1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Angle plots saved" -ForegroundColor Green
    }
    
    # Latent traversals (Figure 1)
    Write-Host "Generating latent traversal visualizations (Figure 1)..."
    python plot_traversals.py --config_id $ConfigId --seed 1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Traversal plots saved" -ForegroundColor Green
    }
    
    Write-Host ""
}

# Main execution
$startTime = Get-Date

if ($Dataset -eq "dsprites" -or $Dataset -eq "both") {
    Run-Evaluation -ConfigId 0 -DatasetName "DSprites"
    Generate-Visualizations -ConfigId 0 -DatasetName "DSprites"
}

if ($Dataset -eq "shapes3d" -or $Dataset -eq "both") {
    Run-Evaluation -ConfigId 1 -DatasetName "3D Shapes"
    Generate-Visualizations -ConfigId 1 -DatasetName "3D Shapes"
}

$endTime = Get-Date
$elapsed = $endTime - $startTime

Write-Host "==========================================" -ForegroundColor Green
Write-Host "ALL EVALUATIONS COMPLETE! 🎉" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Total time: $($elapsed.ToString('hh\:mm\:ss'))"
Write-Host ""
Write-Host "Results saved in Output\ directory"
Write-Host ""
Write-Host "Expected Results (from paper):" -ForegroundColor Cyan
Write-Host "-------------------------------------------"
Write-Host "DSprites:"
Write-Host "  FactorVAE: 79.17 ± 1.64"
Write-Host "  MIG:       0.20 ± 0.02"
Write-Host "  MSE:       2.98 ± 0.28"
Write-Host ""
Write-Host "3D Shapes:"
Write-Host "  FactorVAE: 91.93 ± 3.27"
Write-Host "  MIG:       0.67 ± 0.04"
Write-Host "  MSE:       10.29 ± 0.37"
Write-Host "-------------------------------------------"
Write-Host ""
Write-Host "Compare your results with the expected values above."
Write-Host "Minor variations are expected due to randomness."
Write-Host ""
