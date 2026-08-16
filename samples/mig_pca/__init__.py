"""
MIG (Mutual Information Gap) metric samplers.

Samplers for computing mutual information between latent representations and ground truth factors.
"""

from . import mig_pca_sample_data_dsprites
from . import mig_pca_sample_data_shapes3d

__all__ = [
    'mig_pca_sample_data_dsprites',
    'mig_pca_sample_data_shapes3d',
]
