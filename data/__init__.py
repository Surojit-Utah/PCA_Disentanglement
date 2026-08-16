"""
Data loading module.

Provides dataset loaders for DSprites and 3D Shapes.
"""

from .dsprites_loader import DSpritesLoader
from .shapes3d_loader import Shapes3DLoader

__all__ = [
    'DSpritesLoader',
    'Shapes3DLoader',
]
