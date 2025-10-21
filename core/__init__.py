# ========== core/__init__.py ==========
"""
Ядро приложения - бизнес-логика без зависимости от GUI.
"""

from .data_io import (
    load_dicom_series,
    save_mask_nifti,
    export_history_to_file,
    get_volume_statistics,
    get_mask_statistics
)
from .processing import refine_mask, calculate_mask_difference, apply_preset
from .model_loader import ModelRegistry, discover_models
from .state_manager import UIStateManager, UIState

__all__ = [
    'load_dicom_series',
    'save_mask_nifti',
    'export_history_to_file',
    'get_volume_statistics',
    'get_mask_statistics',
    'refine_mask',
    'calculate_mask_difference',
    'apply_preset',
    'ModelRegistry',
    'discover_models',
    'UIStateManager',
    'UIState'
]
