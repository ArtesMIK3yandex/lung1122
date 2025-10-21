# ========== models/__init__.py ==========
"""
Модули сегментации - плагины для различных моделей.
"""

from .base_model import BaseSegmenter
from .lungmask_segmenter import LungMaskSegmenter

__all__ = [
    'BaseSegmenter',
    'LungMaskSegmenter'
]
