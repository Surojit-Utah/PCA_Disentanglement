"""
3D Shapes dataset loader for disentanglement evaluation.
"""
import os
import numpy as np
from six.moves import range


class Shapes3DLoader:
    """
    Data loader for 3D Shapes dataset.
    
    Environment variable:
        SHAPES3D_DATA_DIR: Path to imgs_train.npz
    """
    
    def __init__(self, latent_dim, batch_size, num_train_data, encoder,
                 use_whiten_data=False, samples_for_global_var=10000):
        """
        Args:
            latent_dim: Dimension of latent representation
            batch_size: Batch size for sampling
            num_train_data: Number of training samples
            encoder: Encoder model for producing latent representations
            use_whiten_data: Whether to whiten the latent representations
            samples_for_global_var: Number of samples for computing whitening matrix
        """
        # Load data from environment variable or default path
        data_dir = os.environ.get('SHAPES3D_DATA_DIR')
        if data_dir is None:
            raise ValueError(
                "SHAPES3D_DATA_DIR environment variable not set. "
                "Please set it to the path of imgs_train.npz"
            )
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"3D Shapes data file not found at: {data_dir}")
        
        self.data_path = data_dir
        self.data = np.load(self.data_path)['arr_0']
        self.images = self.data.astype(np.float32) / 255.0
        # Factor sizes: floor_hue(10), wall_hue(10), object_hue(10), scale(8), shape(4), orientation(15)
        self.factor_sizes = np.array([10, 10, 10, 8, 4, 15])
        self.latent_factor_indices = np.arange(len(self.factor_sizes)).tolist()
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.whiten_matrix = None
        self.seed = 0
        
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_train_data = num_train_data
        self.produce_latent_representation = encoder
        self.whiten_data = use_whiten_data
        
        if self.whiten_data:
            self.compute_global_var(samples_for_global_var)
    
    def whiten(self, X, fudge=1E-04):
        """
        Whiten the data matrix X.
        
        Args:
            X: Data matrix (observations x components)
            fudge: Small constant to avoid numerical issues
            
        Returns:
            X_white: Whitened data
            W: Whitening matrix
        """
        observations = X.shape[0]
        
        # Get the covariance matrix
        Xcov = (np.matmul(X.T, X)) / (observations - 1.0)
        
        # Eigenvalue decomposition of the covariance matrix
        d, V = np.linalg.eigh(Xcov)
        
        # Fudge factor for numerical stability
        D = np.diag(1. / np.sqrt(d + fudge))
        
        # Whitening matrix
        W = np.matmul(np.matmul(V, D), V.T)
        
        # Multiply by the whitening matrix
        X_white = np.matmul(X, W)
        
        return X_white, W
    
    def compute_global_var(self, num_samples):
        """
        Compute global variance and whitening matrix.
        
        Args:
            num_samples: Number of samples to use for computing whitening matrix
        """
        image_indices = np.random.randint(self.images.shape[0], size=num_samples)
        input_images = self.images[image_indices].astype(np.float32)
        latent_representation = self.produce_latent_representation(input_images)
        latent_representation = latent_representation.numpy()
        assert latent_representation.shape[0] == num_samples
        
        _, self.whiten_matrix = self.whiten(latent_representation)
    
    def sample_latent_factors(self, num):
        """
        Sample a batch of latent factors.
        
        Args:
            num: Number of factors to sample
            
        Returns:
            factors: Array of sampled factors (num x num_factors)
        """
        factors = np.zeros(shape=(num, len(self.factor_sizes)), dtype=np.int64)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = np.random.randint(self.factor_sizes[i], size=num)
        
        return factors
    
    def sample_observations_from_factors(self, latent_factors):
        """
        Sample observations given latent factors.
        
        Args:
            latent_factors: Array of latent factors
            
        Returns:
            image_batch: Batch of images
        """
        indices = np.array(np.dot(latent_factors, self.factor_bases), dtype=np.int64)
        image_batch = self.images[indices].astype(np.float32)
        
        return image_batch
    
    def get_train_sample(self):
        """
        Get a training sample with fixed ground truth factor.
        
        Returns:
            representation: Latent representation (whitened if configured)
            gt_factor: Ground truth factor index
        """
        # Sample the GT latent factor (all factors for 3D Shapes)
        gt_factor = np.random.randint(len(self.latent_factor_indices), size=1)[0]
        
        # Sample a batch of latent factors
        latent_factors = self.sample_latent_factors(self.batch_size)
        
        # Fix the selected factor across mini-batch
        latent_factors[:, gt_factor] = latent_factors[0, gt_factor]
        
        # Produce observations
        observation = self.sample_observations_from_factors(latent_factors)
        
        # Produce latent representation
        representation = self.produce_latent_representation(observation).numpy()
        
        # Apply whitening if configured
        if self.whiten_data:
            representation = np.matmul(representation, self.whiten_matrix)
        
        return representation, gt_factor
