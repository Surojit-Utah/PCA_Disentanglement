"""
Data sampling module for disentanglement metrics.

Provides specialized samplers for FactorVAE, MIG, and MSE metrics.
Organized by metric type for clarity.
"""

from .factor_pca import factor_pca_sample_data_dsprites
from .factor_pca import factor_pca_sample_data_shapes3d
from .mig_pca import mig_pca_sample_data_dsprites
from .mig_pca import mig_pca_sample_data_shapes3d
from .mse import sample_data_dsprites
from .mse import sample_data_shapes3d

__all__ = [
    'factor_pca_sample_data_dsprites',
    'factor_pca_sample_data_shapes3d',
    'mig_pca_sample_data_dsprites',
    'mig_pca_sample_data_shapes3d',
    'sample_data_dsprites',
    'sample_data_shapes3d',
]
