"""
FactorVAE metric samplers.

Samplers that fix ground truth factors and vary one at a time for FactorVAE score computation.
"""

from . import factor_pca_sample_data_dsprites
from . import factor_pca_sample_data_shapes3d

__all__ = [
    'factor_pca_sample_data_dsprites',
    'factor_pca_sample_data_shapes3d',
]
