#!/bin/bash
# Reproduce Paper Results
# This script runs all evaluations for AVAE disentanglement analysis
# to reproduce Table 1 and Figures 1-2 from the ICASSP 2024 paper.

set -e  # Exit on error

echo "=========================================="
echo "AVAE Disentanglement Evaluation Pipeline"
echo "=========================================="
echo ""

# Check environment variables
if [ -z "$DSPRITES_DATA_DIR" ]; then
    echo "ERROR: DSPRITES_DATA_DIR environment variable not set"
    echo "Please set: export DSPRITES_DATA_DIR=/path/to/dsprites.npz"
    exit 1
fi

if [ -z "$SHAPES3D_DATA_DIR" ]; then
    echo "ERROR: SHAPES3D_DATA_DIR environment variable not set"
    echo "Please set: export SHAPES3D_DATA_DIR=/path/to/shapes3d.npz"
    exit 1
fi

echo "✓ Environment variables set"
echo "  DSPRITES_DATA_DIR: $DSPRITES_DATA_DIR"
echo "  SHAPES3D_DATA_DIR: $SHAPES3D_DATA_DIR"
echo ""

# Parse command line arguments
DATASET="both"
NUM_SEEDS=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --num-seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset {dsprites|shapes3d|both}  Dataset to evaluate (default: both)"
            echo "  --num-seeds N                       Number of random seeds (default: 10)"
            echo "  --help                              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --dataset shapes3d --num-seeds 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Number of seeds: $NUM_SEEDS"
echo ""

# Function to run evaluation for a dataset
run_evaluation() {
    local config_id=$1
    local dataset_name=$2
    
    echo "=========================================="
    echo "Evaluating $dataset_name"
    echo "=========================================="
    echo ""
    
    for seed in $(seq 1 $NUM_SEEDS); do
        echo "-------------------------------------------"
        echo "Run $seed / $NUM_SEEDS"
        echo "-------------------------------------------"
        
        # FactorVAE Score
        echo "Computing FactorVAE score..."
        python eval/compute_metrics.py \
            --config_id $config_id \
            --metric factor_pca_axis \
            --seed $seed
        
        # MIG Score
        echo "Computing MIG score..."
        python eval/compute_metrics.py \
            --config_id $config_id \
            --metric mig_pca_axis \
            --seed $seed
        
        # MSE
        echo "Computing MSE..."
        python eval/compute_mse.py \
            --config_id $config_id \
            --seed $seed
        
        echo "Run $seed completed ✓"
        echo ""
    done
    
    echo "=========================================="
    echo "$dataset_name evaluation complete!"
    echo "=========================================="
    echo ""

    # Aggregate per-run results into mean +/- std summary files -- required
    # before the summary section below can find anything to display.
    echo "Aggregating results across $NUM_SEEDS runs..."
    python aggregate_results.py --config_id $config_id
    echo ""

    # Print summary statistics
    echo "Results Summary for $dataset_name:"
    echo "-------------------------------------------"
    
    if [ "$dataset_name" == "DSprites" ]; then
        if [ -f "Output/DSprites/Factor_PCA/No_Whiten/factor_pca_axis_metric_stat.txt" ]; then
            echo "FactorVAE Score (Expected: 79.17 ± 1.64):"
            cat "Output/DSprites/Factor_PCA/No_Whiten/factor_pca_axis_metric_stat.txt"
        fi
        if [ -f "Output/DSprites/MIG_PCA/No_Whiten/mig_pca_axis_metric_stat.txt" ]; then
            echo "MIG Score (Expected: 0.20 ± 0.02):"
            cat "Output/DSprites/MIG_PCA/No_Whiten/mig_pca_axis_metric_stat.txt"
        fi
        if [ -f "Output/DSprites/mse_error.txt" ]; then
            echo "MSE (Expected: 2.98 ± 0.28):"
            cat "Output/DSprites/mse_error.txt"
        fi
    else
        if [ -f "Output/Shapes3D/Factor_PCA/No_Whiten/factor_pca_axis_metric_stat.txt" ]; then
            echo "FactorVAE Score (Expected: 91.93 ± 3.27):"
            cat "Output/Shapes3D/Factor_PCA/No_Whiten/factor_pca_axis_metric_stat.txt"
        fi
        if [ -f "Output/Shapes3D/MIG_PCA/No_Whiten/mig_pca_axis_metric_stat.txt" ]; then
            echo "MIG Score (Expected: 0.67 ± 0.04):"
            cat "Output/Shapes3D/MIG_PCA/No_Whiten/mig_pca_axis_metric_stat.txt"
        fi
        if [ -f "Output/Shapes3D/mse_error.txt" ]; then
            echo "MSE (Expected: 10.29 ± 0.37):"
            cat "Output/Shapes3D/mse_error.txt"
        fi
    fi
    
    echo ""
}

# Generate visualizations
generate_visualizations() {
    local config_id=$1
    local dataset_name=$2
    
    echo "=========================================="
    echo "Generating Visualizations for $dataset_name"
    echo "=========================================="
    echo ""
    
    # Angle plots (Figure 2) -- already generated as a side effect of
    # eval/compute_metrics.py (metrics/factor_pca.py and mig_pca.py both call
    # plot_angle_between_pca_axis/plot_histogram_features during the run above),
    # saved per-run under Output/.../Run_N/. No separate script needed.
    echo "Angle distribution plots (Figure 2) already saved per-run during evaluation above."

    # Latent traversals (Figure 1)
    echo "Generating latent traversal visualizations (Figure 1)..."
    python visualization/plot_disentanglement_traversal.py --config_id $config_id --seed 1
    echo "✓ Traversal plots saved"
    
    echo ""
}

# Main execution
start_time=$(date +%s)

if [ "$DATASET" == "dsprites" ] || [ "$DATASET" == "both" ]; then
    run_evaluation 0 "DSprites"
    generate_visualizations 0 "DSprites"
fi

if [ "$DATASET" == "shapes3d" ] || [ "$DATASET" == "both" ]; then
    run_evaluation 1 "3D Shapes"
    generate_visualizations 1 "3D Shapes"
fi

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo "=========================================="
echo "ALL EVALUATIONS COMPLETE! 🎉"
echo "=========================================="
echo ""
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo ""
echo "Results saved in Output/ directory"
echo ""
echo "Expected Results (from paper):"
echo "-------------------------------------------"
echo "DSprites:"
echo "  FactorVAE: 79.17 ± 1.64"
echo "  MIG:       0.20 ± 0.02"
echo "  MSE:       2.98 ± 0.28"
echo ""
echo "3D Shapes:"
echo "  FactorVAE: 91.93 ± 3.27"
echo "  MIG:       0.67 ± 0.04"
echo "  MSE:       10.29 ± 0.37"
echo "-------------------------------------------"
echo ""
echo "Compare your results with the expected values above."
echo "Minor variations are expected due to randomness."
echo ""
