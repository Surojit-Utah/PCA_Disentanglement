"""
Aggregate results from multiple runs to compute statistics.

This script reads individual run results and computes mean ± std statistics
for FactorVAE, MIG, and MSE metrics.
"""

import os
import numpy as np
import argparse
import glob


def aggregate_factor_pca(dataset_name, use_whiten_data):
    """Aggregate FactorVAE scores from multiple runs."""
    if use_whiten_data:
        log_dir = os.path.join('Output', dataset_name, 'Factor_PCA', 'Whiten')
    else:
        log_dir = os.path.join('Output', dataset_name, 'Factor_PCA', 'No_Whiten')
    
    if not os.path.exists(log_dir):
        print(f"Directory not found: {log_dir}")
        return
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(log_dir, 'Run_*'))
    scores = []
    
    for run_dir in sorted(run_dirs):
        output_file = os.path.join(run_dir, 'output.txt')
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    if 'Test accuracy' in line:
                        try:
                            score = float(line.split(':')[1].strip().replace('%', ''))
                            scores.append(score)
                        except:
                            pass
    
    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\nFactorVAE Score ({dataset_name}):")
        print(f"  Runs: {len(scores)}")
        print(f"  Mean: {mean_score:.2f}%")
        print(f"  Std:  {std_score:.2f}%")
        print(f"  Result: {mean_score:.2f} ± {std_score:.2f}")
        
        # Save aggregated result
        stat_file = os.path.join(log_dir, 'factor_pca_axis_metric_stat.txt')
        with open(stat_file, 'w') as f:
            f.write(f'Metric stat using OP PCA vector              : {np.around(mean_score, 2)} ± {np.around(std_score, 2)}\n')
        
        # Save numpy array
        np_file = os.path.join(log_dir, 'factor_pca_axis_metric_stat.npy')
        np.save(np_file, np.array(scores))
        
        print(f"  ✓ Saved to {stat_file}")
    else:
        print(f"No FactorVAE scores found in {log_dir}")


def aggregate_mig_pca(dataset_name, use_whiten_data):
    """Aggregate MIG scores from multiple runs."""
    if use_whiten_data:
        log_dir = os.path.join('Output', dataset_name, 'MIG_PCA', 'Whiten')
    else:
        log_dir = os.path.join('Output', dataset_name, 'MIG_PCA', 'No_Whiten')
    
    if not os.path.exists(log_dir):
        print(f"Directory not found: {log_dir}")
        return
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(log_dir, 'Run_*'))
    scores = []
    
    for run_dir in sorted(run_dirs):
        output_file = os.path.join(run_dir, 'output.txt')
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    if 'MIG Score' in line:
                        try:
                            score = float(line.split(':')[1].strip())
                            scores.append(score)
                        except:
                            pass
    
    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\nMIG Score ({dataset_name}):")
        print(f"  Runs: {len(scores)}")
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Std:  {std_score:.4f}")
        print(f"  Result: {mean_score:.2f} ± {std_score:.2f}")
        
        # Save aggregated result
        stat_file = os.path.join(log_dir, 'mig_pca_axis_metric_stat.txt')
        with open(stat_file, 'w') as f:
            f.write(f'Metric stat using OP PCA vector              : {np.around(mean_score, 2)} ± {np.around(std_score, 2)}\n')
        
        # Save numpy array
        np_file = os.path.join(log_dir, 'mig_pca_axis_metric_stat.npy')
        np.save(np_file, np.array(scores))
        
        print(f"  ✓ Saved to {stat_file}")
    else:
        print(f"No MIG scores found in {log_dir}")


def aggregate_mse(dataset_name):
    """Aggregate MSE scores from multiple runs."""
    log_dir = os.path.join('Output', dataset_name)
    
    if not os.path.exists(log_dir):
        print(f"Directory not found: {log_dir}")
        return
    
    # Find all MSE run files
    mse_files = glob.glob(os.path.join(log_dir, 'mse_run_*.txt'))
    scores = []
    
    for mse_file in sorted(mse_files):
        with open(mse_file, 'r') as f:
            for line in f:
                if 'MSE:' in line:
                    try:
                        score = float(line.split(':')[1].strip())
                        scores.append(score)
                    except:
                        pass
    
    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\nMSE Score ({dataset_name}):")
        print(f"  Runs: {len(scores)}")
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Std:  {std_score:.4f}")
        print(f"  Result: {mean_score:.2f} ± {std_score:.2f}")
        
        # Save aggregated result
        stat_file = os.path.join(log_dir, 'mse_error.txt')
        with open(stat_file, 'w') as f:
            f.write(f'MSE stat : {np.around(mean_score, 2)} ± {np.around(std_score, 2)}\n')
        
        # Save numpy array
        np_file = os.path.join(log_dir, 'mse_error.npy')
        np.save(np_file, np.array(scores))
        
        print(f"  ✓ Saved to {stat_file}")
    else:
        print(f"No MSE scores found in {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate results from multiple evaluation runs")
    parser.add_argument("--config_id", type=int, required=True, help="Configuration ID (0=DSprites, 1=Shapes3D)")
    args = parser.parse_args()

    # Load configuration to get dataset name
    from config.local_config import configurations
    config = configurations[args.config_id]
    dataset_name = config['dataset_name']
    use_whiten_data = config['use_whiten_data']

    print("="*60)
    print(f"Aggregating Results for {dataset_name}")
    print("="*60)

    # Aggregate all metrics
    aggregate_factor_pca(dataset_name, use_whiten_data)
    aggregate_mig_pca(dataset_name, use_whiten_data)
    aggregate_mse(dataset_name)

    print("\n" + "="*60)
    print("✓ Aggregation complete!")
    print("="*60)
