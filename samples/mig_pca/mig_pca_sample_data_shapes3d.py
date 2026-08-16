import numpy as np
import os
np.random.seed(0)
from six.moves import range
from tqdm import tqdm
from sklearn.decomposition import PCA

class sample_data_shapes3d:

  def __init__(self, latent_dim, batch_size, num_train_data, encoder, use_whiten_data, samples_for_global_var):

    self.data_path = os.environ.get('SHAPES3D_DATA_DIR')
    if not self.data_path:
        raise ValueError("SHAPES3D_DATA_DIR environment variable not set. Please set it to the path of imgs_train.npz")
    self.data = np.load(self.data_path)['arr_0']
    self.images = self.data.astype(np.float32) / 255.0
    self.factor_sizes = np.array([10, 10, 10, 8, 4, 15])
    self.latent_factor_indices = np.arange(len(self.factor_sizes)).tolist()
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
    self.whiten_matrix = None

    self.latent_dim = latent_dim
    self.batch_size = batch_size
    self.num_train_data = num_train_data
    self.produce_latent_representation = encoder
    self.whiten_data = use_whiten_data
    if self.whiten_data:
        self.compute_global_var(samples_for_global_var)

  def whiten(self, X, fudge=1E-04):
      # the matrix X should be observations-by-components
      observations = X.shape[0]

      # get the covariance matrix
      Xcov = (np.matmul(X.T, X)) / (observations - 1.0)

      # eigenvalue decomposition of the covariance matrix
      d, V = np.linalg.eigh(Xcov)

      # a fudge factor can be used so that eigenvectors associated with
      # small eigenvalues do not get overamplified.
      D = np.diag(1. / np.sqrt(d + fudge))

      # whitening matrix
      W = np.matmul(np.matmul(V, D), V.T)

      # multiply by the whitening matrix
      X_white = np.matmul(X, W)

      return X_white, W


  def compute_global_var(self, num_samples):
    image_indices = np.random.randint(self.images.shape[0], size=num_samples)
    input_images = self.images[image_indices].astype(np.float32)
    latent_representation = self.produce_latent_representation(input_images)
    latent_representation = latent_representation.numpy()
    assert latent_representation.shape[0] == num_samples

    _, self.whiten_matrix = self.whiten(latent_representation)


  def sample_latent_factors(self, num):
      """Sample a batch of the latent factors."""
      factors = np.zeros(shape=(num, len(self.factor_sizes)), dtype=np.int64)
      for pos, i in enumerate(self.latent_factor_indices):
          factors[:, pos] = np.random.randint(self.factor_sizes[i], size=num)

      return factors


  def sample_observations_from_factors(self, latent_factors):
      """Sample a batch of observations X given a batch of factors Y."""
      indices = np.array(np.dot(latent_factors, self.factor_bases), dtype=np.int64)
      image_batch = self.images[indices].astype(np.float32).astype(np.float32)

      return image_batch


  def generate_batch_factor_code(self, num_points, batch_size=16):
      """Sample a single training sample based on a mini-batch of ground-truth data.

      Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observation as input and
          outputs a representation.
        num_points: Number of points to sample.
        random_state: Numpy random state used for randomness.
        batch_size: Batchsize to sample points.

      Returns:
        representations: Codes (num_codes, num_points)-np array.
        factors: Factors generating the codes (num_factors, num_points)-np array.
      """
      representations = None
      factors = None
      i = 0
      pbar = tqdm(total=num_points)
      while i < num_points:
          num_points_iter = min(num_points - i, batch_size)

          # Sample a batch of latent factors for computing the pairwise difference
          current_factors = self.sample_latent_factors(num_points_iter)

          # Produce observations corresponding to the sampled latent vectors
          current_observations = self.sample_observations_from_factors(current_factors)

          if i == 0:
              # The first factor has only one value. Thus, we ignore that as a ground truth value.
              factors = current_factors
              representations = self.produce_latent_representation(current_observations).numpy()
          else:
              factors = np.vstack((factors, current_factors))
              representations = np.vstack((representations, self.produce_latent_representation(current_observations).numpy()))

          i += num_points_iter
          pbar.update(num_points_iter)
      pbar.close()

      # multiply by the whitening matrix
      if self.whiten_data:
          representations = np.matmul(representations, self.whiten_matrix)

      return np.transpose(representations), np.transpose(factors)


  def get_data(self, num_samples):
      observations, gt_factors = self.generate_batch_factor_code(num_samples)

      return gt_factors, observations


  def get_train_sample_pca_axis(self):
      # sample the GT latent factor
      gt_factor = np.random.randint(len(self.latent_factor_indices), size=1)[0]

      # Sample a batch of latent factors for computing the pairwise difference
      latent_factors = self.sample_latent_factors(self.batch_size)

      # Fix the selected factor across mini-batch.
      latent_factors[:, gt_factor] = latent_factors[0, gt_factor]

      # Produce observations corresponding to the sampled latent vectors
      observation = self.sample_observations_from_factors(latent_factors)

      # Produce latent representation of the sampled images
      representation = self.produce_latent_representation(observation).numpy()

      # multiply by the whitening matrix
      if self.whiten_data:
          pca_representation = np.matmul(representation, self.whiten_matrix)
      else:
          pca_representation = representation

      # PCA projection using Scipy
      pca = PCA()
      pca.fit(pca_representation)
      vectors = pca.components_
      explained_variances = pca.explained_variance_
      gt_vector = vectors[np.argmin(explained_variances)]

      return gt_factor, gt_vector


  def get_data_pca_axis(self, num_samples):
      gt_factor = np.zeros((num_samples), dtype=np.uint8)
      gt_factor_pca_axis = np.zeros((num_samples, self.latent_dim))
      for sample_index in tqdm(range(num_samples)):
          factor_index, gt_pca_axis = self.get_train_sample_pca_axis()
          gt_factor[sample_index] = factor_index
          gt_factor_pca_axis[sample_index, :] = gt_pca_axis

      return gt_factor, gt_factor_pca_axis