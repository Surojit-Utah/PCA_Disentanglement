import numpy as np
from six.moves import range
from sklearn.decomposition import PCA
import scipy.spatial.distance as distance
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os


class factor_pca_axis:

  def __init__(self, dataset_name, use_whiten_data, run_id, encoder):

    self.dataset_name = dataset_name
    self.run_id = run_id
    self.produce_latent_representation = encoder
    if use_whiten_data:
        self.savedir = os.path.join('Output', dataset_name, 'Factor_PCA', 'Whiten', 'Run_' + str(self.run_id))
    else:
        self.savedir = os.path.join('Output', dataset_name, 'Factor_PCA', 'No_Whiten', 'Run_' + str(self.run_id))
    if not os.path.isdir(self.savedir):
        os.makedirs(self.savedir, exist_ok=True)
    self.runfileptr = open(os.path.join(self.savedir, 'output.txt'), 'w')


  def plot_angle_between_pca_axis(self, plot_array, method='mean_pca_vector'):
      plot_array = np.around(plot_array, 2)
      fig, ax = plt.subplots()
      im = ax.imshow(plot_array)

      # Loop over data dimensions and create text annotations.
      for i in range(plot_array.shape[0]):
          for j in range(plot_array.shape[1]):
              text = ax.text(j, i, plot_array[i, j], ha="center", va="center", color="r")

      if method == 'mean_pca_vector':
          pca_angle_path = os.path.join(self.savedir, 'mean_angle_between_pca_axis.png')
          ax.set_title("Angle between the principal components.")
      elif method == 'mean_outer_prod':
          pca_angle_path = os.path.join(self.savedir, 'matrix_mean_angle_between_pca_axis.png')
          ax.set_title("Angle between the principal components.")
      elif method == 'mutual_entropy':
          pca_angle_path = os.path.join(self.savedir, 'mutual_entropy.png')
          ax.set_title("Entropy between the latent code and GT.")

      fig.tight_layout()
      plt.savefig(pca_angle_path)
      plt.close(plt.gcf())


  def plot_histogram_features(self, plot_array, feature_index):
      # print("Total number of entries for histogram plot : " + str(len(plot_array)))

      # Latent distance
      fig, ax1 = plt.subplots()
      n_bins = 200
      ax1.hist(plot_array, bins=n_bins, histtype='stepfilled', color='b')
      ax1.set_ylabel('sample_count', color='b')

      pca_angle_path = os.path.join(self.savedir, 'angle_distribution_between_pca_axis_' + str(feature_index) + '.png')
      plt.savefig(pca_angle_path)
      plt.close(plt.gcf())


  def get_pca_axis_using_outer_prod(self, sampled_data=None):

      if sampled_data is not None:
          print("Using the sampled data for deriving the PCA axis....")
          gt_factor, gt_factor_pca_axis = sampled_data[0], sampled_data[1]

      unique_values = np.unique(gt_factor)
      mean_pca_axes = np.zeros((len(unique_values), gt_factor_pca_axis.shape[1]))
      gt_index = 0
      for unique_val in unique_values.tolist():
          cur_gt_factor_indices = np.where(gt_factor == unique_val)[0]
          cur_gt_factor_pca_axis = gt_factor_pca_axis[cur_gt_factor_indices]
          pairwise_distance_gt_factor_pca_axis = distance.squareform(
              distance.pdist(cur_gt_factor_pca_axis, metric='cosine'))
          pairwise_distance_gt_factor_pca_axis = np.sqrt(np.power(-pairwise_distance_gt_factor_pca_axis + 1, 2))

          # Histogram of the pairwise angle between the PCA axes
          self.plot_histogram_features(np.rad2deg(np.arccos(pairwise_distance_gt_factor_pca_axis.flatten())),
                                       unique_val)

          collection_pca_mat = np.zeros((cur_gt_factor_pca_axis.shape[1], cur_gt_factor_pca_axis.shape[1]))
          for pca_axis_index in range(cur_gt_factor_pca_axis.shape[0]):
              cur_pca_axis = cur_gt_factor_pca_axis[pca_axis_index]

              collection_pca_mat += np.outer(cur_pca_axis, cur_pca_axis.T)

          # Average matrix for PCA decomposition
          # This gives us the representative PCA vector for a single GT factor
          collection_pca_mat /= cur_gt_factor_pca_axis.shape[0]
          pca = PCA()
          pca.fit(collection_pca_mat)
          vectors = pca.components_
          explained_variances = pca.explained_variance_
          # Store the mean PCA vector for all the GT factors
          mean_pca_axes[gt_index] = vectors[np.argmax(explained_variances)]
          gt_index += 1

      pairwise_distance_pca_axis = distance.squareform(distance.pdist(mean_pca_axes, metric='cosine'))
      pairwise_distance_pca_axis = -pairwise_distance_pca_axis + 1
      pairwise_angle_pca_axis = np.rad2deg(np.arccos(pairwise_distance_pca_axis))
      self.plot_angle_between_pca_axis(pairwise_angle_pca_axis, method='mean_outer_prod')

      return mean_pca_axes


  def get_model_prediction(self, data, mean_pca_axes, ignore_first_index):

      pairwise_distance_pca_axis = distance.cdist(data, mean_pca_axes, metric='cosine')
      pairwise_distance_pca_axis = np.power(-pairwise_distance_pca_axis + 1, 2)
      # Dsprites data set
      if ignore_first_index:
          prediction = np.argmax(pairwise_distance_pca_axis, axis=1) + 1
      # Shapes3D data set
      else:
          prediction = np.argmax(pairwise_distance_pca_axis, axis=1)

      return prediction