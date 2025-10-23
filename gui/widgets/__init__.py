# gui/widgets/__init__.py

from .status_widget import StatusWidget
from .segmentation_widget import SegmentationWidget
from .processing_widget import ProcessingWidget
from .save_widget import SaveWidget
from .admin_panel import AdminPanel
from .projection_manager import ProjectionManager
from .viewer_widget import ViewerWidget

__all__ = [
    'StatusWidget',
    'SegmentationWidget',
    'ProcessingWidget',
    'SaveWidget',
    'AdminPanel',
    'ProjectionManager',
    'ViewerWidget',
]
