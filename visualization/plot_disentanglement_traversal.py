import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import sys
import numpy as np

# Add parent directory to path if running from visualization subdirectory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from samples import factor_pca_sample_data_dsprites, mig_pca_sample_data_dsprites
from samples import factor_pca_sample_data_shapes3d, mig_pca_sample_data_shapes3d
from models.ae_model_dsprites import Encoder as DSpritesEncoder, Decoder as DSpritesDecoder
from models.ae_model_shapes3d import Encoder as Shapes3DEncoder, Decoder as Shapes3DDecoder
from metrics.mig_pca import mig_pca_axis
from config.local_config import configurations
from utils.gpu_utils import select_GPU, set_seed
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse


def show_images(images, row_cnt, col_cnt):

    fig = plt.figure(figsize=(col_cnt, row_cnt))
    grid_spec = gridspec.GridSpec(ncols=col_cnt, nrows=row_cnt, figure=fig)
    grid_spec.update(wspace=0.05, hspace=0.05)

    for i in range(row_cnt):
        for j in range(col_cnt):

            img_index = i*col_cnt + j
            img = images[img_index, :, :, :]

            ax = fig.add_subplot(grid_spec[i, j])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.axis('off')
            plt.imshow(img, vmin=0, vmax= 255)

    return fig


if __name__=="__main__":

    use_gpu, mem_free = select_GPU()
    print("Selected GPU for training : " + str(use_gpu) + " with available memory : " + str(mem_free//(1024*1024*1024)))

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1, help="Random seed / Run ID")
    args = parser.parse_args()
    config = configurations[args.config_id]

    # model configurations
    model_name                  = config['model_name']
    dataset_name                = config['dataset_name']
    batch_size                  = config['train_data_size']
    latent_dim                  = config['latent_dim']
    num_filter                  = config['num_filter']
    samples_for_global_var      = config['samples_for_global_var']
    encoder_use_batch_norm      = config['encoder_use_batch_norm']
    decoder_use_batch_norm      = config['decoder_use_batch_norm']
    num_train_data              = config['num_train_data']
    num_eval_data               = config['num_eval_data']
    use_whiten_data             = config['use_whiten_data']
    num_bins                    = config['num_bins']
    ori_bandwidth               = config['ori_bandwidth']
    alpha                       = np.sqrt(1/(1+ori_bandwidth**2)).astype(np.float32)
    bandwidth                   = ori_bandwidth*alpha
    Guassian_Prior_Std_Dev      = np.sqrt(1 - bandwidth**2).astype(np.float32)
    savedir                     = os.path.join('logs', dataset_name)
    eval_ids                    = [args.seed]
    enable_debug_mode           = False


    for run_id in eval_ids:
        set_seed(run_id-1)

        # load checkpoint
        if dataset_name == 'DSprites':
            encoder = DSpritesEncoder(latent_dim=latent_dim, num_filter=num_filter)
            decoder = DSpritesDecoder(latent_dim=latent_dim, num_filter=num_filter)
        elif 'Shapes3D' in dataset_name:
            encoder = Shapes3DEncoder(latent_dim=latent_dim, num_filter=num_filter)
            decoder = Shapes3DDecoder(latent_dim=latent_dim, num_filter=num_filter)
        learning_rate = 5e-4
        optimizer = Adam(learning_rate)
        checkpoint_dir = config['model_checkpoint_dir'].format(run_id=run_id)
        model_checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        status = model_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        status.assert_existing_objects_matched()
        print("Loaded saved model parameters!!")

        if dataset_name == 'DSprites':
            sample_data_obj = mig_pca_sample_data_dsprites.sample_data_dsprites(latent_dim, batch_size,
                                                                                num_train_data, encoder,
                                                                                use_whiten_data,
                                                                                samples_for_global_var)
        elif 'Shapes3D' in dataset_name:
            sample_data_obj = mig_pca_sample_data_shapes3d.sample_data_shapes3d(latent_dim, batch_size,
                                                                                num_train_data, encoder,
                                                                                use_whiten_data,
                                                                                samples_for_global_var)

        # get data for MIG evaluation
        gt_factor, latent_representation = sample_data_obj.get_data(num_train_data)
        if enable_debug_mode:
            mean_latent_vector = np.mean(latent_representation, axis=0)
            std_latent_vector = np.std(latent_representation, axis=0)
            norm_latent_representation = np.linalg.norm(latent_representation, axis=1)
            mean_norm_latent_representation = np.mean(norm_latent_representation)
            std_norm_latent_representation = np.std(norm_latent_representation)
            print(mean_latent_vector, std_latent_vector)
            print(mean_norm_latent_representation, std_norm_latent_representation)
            input()

        # Get data for determining the PCA axis
        pca_gt_factor, pca_latent_representation = sample_data_obj.get_data_pca_axis(num_train_data)

        # Get the PCA axis onto which the sampled data will be projected
        print("Using the Eigen decomposition of the outer product of PCA vectors for a latent GT factor")
        mig_pca_axis_obj = mig_pca_axis(dataset_name, use_whiten_data, run_id, encoder)
        op_pca_vectors = mig_pca_axis_obj.get_pca_axis_using_outer_prod((pca_gt_factor, pca_latent_representation))

        # save images
        generated_image_dir = os.path.join(mig_pca_axis_obj.savedir, "Traversals_" + str(run_id))
        if not os.path.isdir(generated_image_dir):
            os.makedirs(generated_image_dir, exist_ok=True)
        num_of_stddev = 1
        num_images = 20
        repeat_an_axis = 10
        proj_gt_factor_pca_axis = np.dot(latent_representation, op_pca_vectors.T)
        if enable_debug_mode:
            mean_proj_gt_factor_pca_axis = np.mean(proj_gt_factor_pca_axis, axis=0)
            std_proj_gt_factor_pca_axis = np.std(proj_gt_factor_pca_axis, axis=0)
            norm_proj_gt_factor_pca_axis = np.linalg.norm(proj_gt_factor_pca_axis, axis=1)
            mean_norm_proj_gt_factor_pca_axis = np.mean(norm_proj_gt_factor_pca_axis)
            std_norm_proj_gt_factor_pca_axis = np.std(norm_proj_gt_factor_pca_axis)
            print(mean_proj_gt_factor_pca_axis, std_proj_gt_factor_pca_axis)
            print(mean_norm_proj_gt_factor_pca_axis, std_norm_proj_gt_factor_pca_axis)
            input()

        for vector_index in range(latent_dim):
            save_gen_images = None
            for repeat_index in range(repeat_an_axis):

                # When using the mean vector in the transformed space, for setting the variable latent factors
                # pca_sample_array = np.expand_dims(np.mean(proj_gt_factor_pca_axis, axis=0), axis=0) + \
                #                    np.expand_dims(np.random.standard_normal(latent_dim)*(Guassian_Prior_Std_Dev*0.1), axis=0)

                # Use the learned representation of an input
                pca_sample_array = np.expand_dims(proj_gt_factor_pca_axis[repeat_index], axis=0)
                pca_space_samples = np.repeat(pca_sample_array, num_images, axis=0)
                pca_axis_values = np.linspace(-num_of_stddev*Guassian_Prior_Std_Dev, num_of_stddev*Guassian_Prior_Std_Dev, num_images)
                pca_space_samples[:, vector_index] += pca_axis_values
                if enable_debug_mode:
                    mean_pca_space_samples = np.mean(pca_space_samples, axis=0)
                    std_pca_space_samples = np.std(pca_space_samples, axis=0)
                    norm_pca_space_samples = np.linalg.norm(pca_space_samples, axis=1)
                    mean_norm_pca_space_samples = np.mean(norm_pca_space_samples)
                    std_norm_pca_space_samples = np.std(norm_pca_space_samples)
                    print(pca_sample_array)
                    print(pca_axis_values)
                    print(pca_space_samples)
                    print(mean_pca_space_samples, std_pca_space_samples)
                    print(mean_norm_pca_space_samples, std_norm_pca_space_samples)
                    input()
                # When using the mean vector in the transformed space, add the mean vector to the inverse transformation
                # data_space_proj = np.dot(pca_space_samples, op_pca_vectors) + mean_latent_vector
                data_space_proj = np.dot(pca_space_samples, op_pca_vectors)
                if enable_debug_mode:
                    mean_data_space_proj = np.mean(data_space_proj, axis=0)
                    std_data_space_proj = np.std(data_space_proj, axis=0)
                    norm_data_space_proj = np.linalg.norm(data_space_proj, axis=1)
                    mean_norm_data_space_proj = np.mean(norm_data_space_proj)
                    std_norm_data_space_proj = np.std(norm_data_space_proj)
                    print(mean_data_space_proj, std_data_space_proj)
                    print(mean_norm_data_space_proj, std_norm_data_space_proj)
                    input()
                Gen_Test_Images = decoder(data_space_proj, use_batch_norm=decoder_use_batch_norm,
                                          training=False).numpy()
                Gen_Test_Images = (Gen_Test_Images * 255.0).astype(np.uint8)
                if save_gen_images is None:
                    save_gen_images = np.zeros((repeat_an_axis * num_images, Gen_Test_Images.shape[1],
                                                Gen_Test_Images.shape[2], Gen_Test_Images.shape[3]), dtype=np.uint8)
                    print("Created an array of generated images....")
                    print(save_gen_images.shape)
                save_gen_images[repeat_index * num_images:(repeat_index + 1) * num_images] = Gen_Test_Images
            gen_test_image_path = os.path.join(generated_image_dir,
                                               "disentanglement_study_pca_" + str(vector_index) + ".png")
            fig = show_images(save_gen_images, repeat_index, num_images)
            plt.savefig(gen_test_image_path)
            plt.close(plt.gcf())