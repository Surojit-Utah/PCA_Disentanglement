"""
Compute MSE (Mean Squared Error) reconstruction metric for AVAE models.

This script evaluates the reconstruction quality of trained AVAE models.
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

from samples import sample_data_dsprites, sample_data_shapes3d
from models import ae_model_dsprites, ae_model_shapes3d
from config.local_config import configurations
from utils.gpu_utils import select_GPU, set_seed
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


@tf.function
def autoencoder_loss(x, x_hat):
    """Compute MSE reconstruction loss."""
    batch_size = tf.shape(x)[0]
    inputs = tf.reshape(x, (batch_size, -1))
    x_hat = tf.reshape(x_hat, (batch_size, -1))
    reconstruction = tf.reduce_mean(tf.reduce_sum((inputs - x_hat)**2, 1))
    return reconstruction


def show_combined_images(images, trans_images, row_cnt, col_cnt):
    """Show original and reconstructed images side by side."""
    fig = plt.figure()
    grid_spec = gridspec.GridSpec(ncols=col_cnt, nrows=row_cnt, figure=fig)
    grid_spec.update(wspace=0.05, hspace=0.05)

    for i in range(row_cnt):
        for j in range(0, col_cnt, 2):
            img_index = i * row_cnt + (j // 2)

            img = images[img_index, :, :, :]
            trans_img = trans_images[img_index, :, :, :]

            img = (img * 255.0).astype(np.uint8)
            trans_img = (trans_img * 255.0).astype(np.uint8)

            # Clipping the Range [0, 255]
            img = np.clip(img, 0, 255)
            trans_img = np.clip(trans_img, 0, 255)

            ax = fig.add_subplot(grid_spec[i, j])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.axis('off')
            plt.imshow(img, vmin=0, vmax=255)

            ax = fig.add_subplot(grid_spec[i, j + 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.axis('off')
            plt.imshow(trans_img, vmin=0, vmax=255)

    return fig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute MSE reconstruction metric for AVAE models")
    parser.add_argument("--config_id", type=int, required=True, help="Configuration ID (0=DSprites, 1=Shapes3D)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed / Run ID")
    args = parser.parse_args()

    # Select GPU
    use_gpu, mem_free = select_GPU()
    print(f"Selected GPU {use_gpu} with {mem_free//(1024*1024*1024)} GB available memory")

    # Load configuration
    config = configurations[args.config_id]
    run_id = args.seed

    # Model configurations
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    batch_size = config['train_data_size']
    latent_dim = config['latent_dim']
    num_filter = config['num_filter']
    samples_for_global_var = config['samples_for_global_var']
    encoder_use_batch_norm = config['encoder_use_batch_norm']
    decoder_use_batch_norm = config['decoder_use_batch_norm']
    num_train_data = config['num_train_data']
    num_eval_data = config['num_eval_data']
    use_whiten_data = config['use_whiten_data']
    num_bins = config['num_bins']
    
    # Model checkpoint directory (from configuration)
    checkpoint_dir = config['model_checkpoint_dir'].format(run_id=run_id)
    
    # Output directory
    log_dir = os.path.join('Output', dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Run ID: {run_id}")
    print(f"  Checkpoint: {checkpoint_dir}\n")

    # Load checkpoint
    if 'DSprites' in dataset_name:
        encoder = ae_model_dsprites.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_dsprites.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        sample_data_obj = sample_data_dsprites.sample_data_dsprites(latent_dim, batch_size, num_train_data)
    elif 'Shapes3D' in dataset_name:
        encoder = ae_model_shapes3d.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_shapes3d.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        sample_data_obj = sample_data_shapes3d.sample_data_shapes3d(latent_dim, batch_size, num_train_data)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    learning_rate = 5e-4
    optimizer = Adam(learning_rate)
    model_checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    status = model_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    status.assert_existing_objects_matched()
    print("✓ Loaded saved model parameters\n")

    # Get evaluation data
    print("Sampling evaluation data...")
    input_data = sample_data_obj.get_data(num_train_data)
    max_iter = num_train_data // batch_size

    # Compute MSE
    print("Computing MSE reconstruction error...")
    avg_recons_loss = 0
    save_a_batch = False
    
    for iter_index in range(max_iter):
        start_index = iter_index * batch_size
        end_index = iter_index * batch_size + batch_size
        input_batch = input_data[start_index:end_index]

        # Reconstruction of data
        encoder_output = encoder(input_batch, use_batch_norm=encoder_use_batch_norm, training=False)
        decoder_output = decoder(encoder_output, use_batch_norm=decoder_use_batch_norm, training=False)

        # DSprites decoder produces logits
        if 'DSprites' in dataset_name:
            decoder_output = tf.math.sigmoid(decoder_output)

        # MSE error
        avg_recons_loss += autoencoder_loss(input_batch, decoder_output).numpy()
        
        # Save a batch of reconstructed images (only once)
        if not save_a_batch:
            row_cnt = col_cnt = int(np.sqrt(batch_size))
            fig = show_combined_images(input_batch, decoder_output.numpy(), row_cnt, col_cnt * 2)
            reconstructed_image_path = os.path.join(log_dir, f'recons_example_run_{run_id}.png')
            plt.savefig(reconstructed_image_path)
            plt.close(plt.gcf())
            save_a_batch = True
            print(f"✓ Saved reconstruction examples to {reconstructed_image_path}")

    avg_recons_loss /= max_iter
    
    print("\n" + "="*50)
    print(f"MSE Reconstruction Error: {avg_recons_loss:.4f}")
    print("="*50 + "\n")

    # Save individual run result
    run_filepath = os.path.join(log_dir, f'mse_run_{run_id}.txt')
    with open(run_filepath, 'w') as f:
        f.write(f'Run {run_id} MSE: {avg_recons_loss:.4f}\n')
    
    print(f"✓ Results saved to {log_dir}")
    print("\n✓ Computation complete!")
