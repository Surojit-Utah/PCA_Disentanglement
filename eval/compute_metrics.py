"""
Compute disentanglement metrics (FactorVAE and MIG) for AVAE models.

This script evaluates trained AVAE models using PCA-based disentanglement metrics.
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import sys
import numpy as np

# Add parent directory to path to import modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Change working directory to parent so Output/ is created in the right place
os.chdir(parent_dir)

from samples import factor_pca_sample_data_dsprites, mig_pca_sample_data_dsprites
from samples import factor_pca_sample_data_shapes3d, mig_pca_sample_data_shapes3d
from models import ae_model_dsprites, ae_model_shapes3d
from metrics import factor_pca_axis, mig_pca_axis
from config.local_config import configurations
from sklearn.metrics import accuracy_score
from utils.gpu_utils import select_GPU, set_seed
import argparse


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Compute disentanglement metrics for AVAE models")
    parser.add_argument("--config_id", type=int, required=True, help="Configuration ID (0=DSprites, 1=Shapes3D)")
    parser.add_argument("--metric", type=str, required=True, help="Metric type: factor_pca_axis or mig_pca_axis")
    parser.add_argument("--seed", type=int, default=1, help="Random seed / Run ID")
    args = parser.parse_args()

    # Select GPU
    use_gpu, mem_free = select_GPU()
    print(f"Selected GPU {use_gpu} with {mem_free//(1024*1024*1024)} GB available memory")

    # Load configuration
    config = configurations[args.config_id]
    metric_type = args.metric
    run_id = args.seed

    # Set seed
    set_seed(run_id - 1)

    # Model configurations
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    batch_size = config['train_data_size']
    latent_dim = config['latent_dim']
    num_filter = config['num_filter']
    samples_for_global_var = config['samples_for_global_var']
    encoder_use_batch_norm = config['encoder_use_batch_norm']
    num_train_data = config['num_train_data']
    num_eval_data = config['num_eval_data']
    use_whiten_data = config['use_whiten_data']
    num_bins = config['num_bins']
    
    # Model checkpoint directory (from configuration)
    checkpoint_dir = config['model_checkpoint_dir'].format(run_id=run_id)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Metric: {metric_type}")
    print(f"  Run ID: {run_id}")
    print(f"  Checkpoint: {checkpoint_dir}\n")

    # Load checkpoint
    if dataset_name == 'DSprites':
        encoder = ae_model_dsprites.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_dsprites.Decoder(latent_dim=latent_dim, num_filter=num_filter)
    elif 'Shapes3D' in dataset_name:
        encoder = ae_model_shapes3d.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_shapes3d.Decoder(latent_dim=latent_dim, num_filter=num_filter)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    learning_rate = 5e-4
    optimizer = Adam(learning_rate)
    model_checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    status = model_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    status.assert_existing_objects_matched()
    print("✓ Loaded saved model parameters\n")

    if metric_type == 'factor_pca_axis':
        
        ignore_first_index = (dataset_name == 'DSprites')
        
        # Create data sampler
        if dataset_name == 'DSprites':
            sample_data_obj = factor_pca_sample_data_dsprites.sample_data_dsprites(
                latent_dim, batch_size, num_train_data, encoder, 
                use_whiten_data, samples_for_global_var)
        elif 'Shapes3D' in dataset_name:
            sample_data_obj = factor_pca_sample_data_shapes3d.sample_data_shapes3d(
                latent_dim, batch_size, num_train_data, encoder, 
                use_whiten_data, samples_for_global_var)
        
        # Get training data
        print("Sampling training data...")
        gt_factor, gt_factor_pca_axis = sample_data_obj.get_data(num_train_data)
        
        # Get the mean PCA vectors using the train data
        print("Computing PCA-based factor directions...")
        factor_pca_axis_obj = factor_pca_axis.factor_pca_axis(
            dataset_name, use_whiten_data, run_id, encoder)
        avg_pca_vectors_using_outer_prod = factor_pca_axis_obj.get_pca_axis_using_outer_prod(
            sampled_data=(gt_factor, gt_factor_pca_axis))
        
        # Get evaluation data
        print("Sampling evaluation data...")
        eval_gt_factor, eval_gt_factor_pca_axis = sample_data_obj.get_data(num_eval_data)
        
        print("\n" + "="*50)
        print("Evaluate FactorVAE Score")
        print("="*50)
        factor_pca_axis_obj.runfileptr.write("Evaluate accuracy\n")
        factor_pca_axis_obj.runfileptr.write("=================\n")
        
        # Training accuracy
        train_classification = factor_pca_axis_obj.get_model_prediction(
            gt_factor_pca_axis, avg_pca_vectors_using_outer_prod, ignore_first_index)
        train_accuracy = accuracy_score(gt_factor, train_classification) * 100
        print(f"Training accuracy: {train_accuracy:.2f}%")
        factor_pca_axis_obj.runfileptr.write(f"Training accuracy: {train_accuracy:.2f}%\n")
        
        # Test accuracy
        test_classification = factor_pca_axis_obj.get_model_prediction(
            eval_gt_factor_pca_axis, avg_pca_vectors_using_outer_prod, ignore_first_index)
        test_accuracy = accuracy_score(eval_gt_factor, test_classification) * 100
        print(f"Test accuracy: {test_accuracy:.2f}%")
        print("="*50 + "\n")
        factor_pca_axis_obj.runfileptr.write(f"Test accuracy: {test_accuracy:.2f}%\n\n")
        factor_pca_axis_obj.runfileptr.flush()
        factor_pca_axis_obj.runfileptr.close()
        
        print(f"✓ Results saved to {factor_pca_axis_obj.savedir}")
    
    elif metric_type == 'mig_pca_axis':
        
        # Create data sampler
        if dataset_name == 'DSprites':
            sample_data_obj = mig_pca_sample_data_dsprites.sample_data_dsprites(
                latent_dim, batch_size, num_train_data, encoder, 
                use_whiten_data, samples_for_global_var)
        elif 'Shapes3D' in dataset_name:
            sample_data_obj = mig_pca_sample_data_shapes3d.sample_data_shapes3d(
                latent_dim, batch_size, num_train_data, encoder, 
                use_whiten_data, samples_for_global_var)
        
        # Get data for determining the PCA axis
        print("Sampling data for PCA axis determination...")
        pca_gt_factor, pca_latent_representation = sample_data_obj.get_data_pca_axis(num_train_data)
        
        # Get data for MIG evaluation
        print("Sampling data for MIG evaluation...")
        gt_factor, latent_representation = sample_data_obj.get_data(num_train_data)
        
        # Get the PCA axis onto which the sampled data will be projected
        print("Computing PCA-based MIG directions...")
        mig_pca_axis_obj = mig_pca_axis.mig_pca_axis(
            dataset_name, use_whiten_data, run_id, encoder)
        
        print("\n" + "="*50)
        print("Evaluate MIG Score")
        print("="*50)
        mig_pca_axis_obj.runfileptr.write("Evaluate accuracy\n")
        mig_pca_axis_obj.runfileptr.write("=================\n")
        
        # Compute MIG score
        op_pca_vectors = mig_pca_axis_obj.get_pca_axis_using_outer_prod(
            (pca_gt_factor, pca_latent_representation))
        proj_gt_factor_pca_axis = np.matmul(op_pca_vectors, latent_representation)
        mig_score = mig_pca_axis_obj.compute_mig(proj_gt_factor_pca_axis, gt_factor, num_bins)
        
        print(f"MIG Score: {mig_score:.4f}")
        print("="*50 + "\n")
        mig_pca_axis_obj.runfileptr.write(
            f"MIG Score (PCA-based): {mig_score:.4f}\n")
        mig_pca_axis_obj.runfileptr.flush()
        mig_pca_axis_obj.runfileptr.close()
        
        print(f"✓ Results saved to {mig_pca_axis_obj.savedir}")
    
    else:
        raise ValueError(f"Unknown metric type: {metric_type}. Use 'factor_pca_axis' or 'mig_pca_axis'")
    
    print("\n✓ Computation complete!")
