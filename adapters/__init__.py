"""
Model adapters for different diffusion model types
"""

from .image_adapter import ImageDiffusionAdapter, FluxAdapter, SDXLAdapter
from .video_adapter import VideoDiffusionAdapter, AnimateDiffAdapter

__all__ = [
    'ImageDiffusionAdapter',
    'FluxAdapter', 
    'SDXLAdapter',
    'VideoDiffusionAdapter',
    'AnimateDiffAdapter'
]