import numpy as np
import os
np.random.seed(0)
from six.moves import range
from tqdm import tqdm
from sklearn.decomposition import PCA

class sample_data_dsprites:

  def __init__(self, latent_dim, batch_size, num_train_data):

    self.data_path = os.environ.get('DSPRITES_DATA_DIR')
    if not self.data_path:
        raise ValueError("DSPRITES_DATA_DIR environment variable not set. Please set it to the path of dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    self.data = np.load(self.data_path, encoding="latin1", allow_pickle=True)
    self.images = np.array(self.data["imgs"])
    self.factor_sizes = np.array(self.data["metadata"][()]["latents_sizes"], dtype=np.int64)
    self.latent_factor_indices = np.arange(len(self.factor_sizes)).tolist()
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)

    self.latent_dim = latent_dim
    self.batch_size = batch_size
    self.num_train_data = num_train_data


  def sample_latent_factors(self, num):
      """Sample a batch of the latent factors."""
      factors = np.zeros(shape=(num, len(self.factor_sizes)), dtype=np.int64)
      for pos, i in enumerate(self.latent_factor_indices):
          factors[:, pos] = np.random.randint(self.factor_sizes[i], size=num)

      return factors


  def sample_observations_from_factors(self, latent_factors):
      """Sample a batch of observations X given a batch of factors Y."""
      indices = np.array(np.dot(latent_factors, self.factor_bases), dtype=np.int64)
      image_batch = np.expand_dims(self.images[indices].astype(np.float32), axis=-1).astype(np.float32)

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
        sampled_data: Codes (num_codes, num_points)-np array.
        factors: Factors generating the codes (num_factors, num_points)-np array.
      """
      sampled_data = None
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
              factors = current_factors[:, 1:]
              sampled_data = current_observations
          else:
              factors = np.vstack((factors, current_factors[:, 1:]))
              sampled_data = np.vstack((sampled_data, current_observations))

          i += num_points_iter
          pbar.update(num_points_iter)
      pbar.close()

      return sampled_data


  def get_data(self, num_samples):
      observations = self.generate_batch_factor_code(num_samples)

      return observations