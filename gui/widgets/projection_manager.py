"""
Менеджер окон проекций - динамическая система отображения срезов.
Легко добавлять/удалять проекции без изменения MainWindow.
"""

from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter
import numpy as np


class ProjectionView(QLabel):
    """Виджет отображения одной проекции"""
    
    slice_changed = pyqtSignal(int)  # Сигнал изменения среза
    
    def __init__(self, orientation: str, parent=None):
        super().__init__(parent)
        self.orientation = orientation  # 'axial', 'sagittal', 'coronal'
        self.current_slice = 0
        self.max_slices = 0
        self.image_data = None
        
        # Настройки отображения (Window/Level)
        self.window_center = 40
        self.window_width = 400
        
        # Навигация
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: black; border: 1px solid #555; }")
        self.setText(f"{orientation.upper()}\n(Нет данных)")
        self.setStyleSheet("""
            QLabel {
                background-color: black;
                color: #888;
                border: 2px solid #555;
                font-size: 14px;
            }
        """)
    
    def set_data(self, volume: np.ndarray):
        """Устанавливает 3D-объем для отображения"""
        self.image_data = volume
        
        if self.orientation == 'axial':
            self.max_slices = volume.shape[0]
            self.current_slice = volume.shape[0] // 2
        elif self.orientation == 'sagittal':
            self.max_slices = volume.shape[2]
            self.current_slice = volume.shape[2] // 2
        elif self.orientation == 'coronal':
            self.max_slices = volume.shape[1]
            self.current_slice = volume.shape[1] // 2
        
        self.update_display()
    
    def get_current_slice_data(self) -> np.ndarray:
        """Возвращает текущий срез"""
        if self.image_data is None:
            return None
        
        try:
            if self.orientation == 'axial':
                return self.image_data[self.current_slice, :, :]
            elif self.orientation == 'sagittal':
                return self.image_data[:, :, self.current_slice]
            elif self.orientation == 'coronal':
                return self.image_data[:, self.current_slice, :]
        except IndexError:
            return None
    
    def set_slice(self, slice_idx: int):
        """Устанавливает текущий срез"""
        if 0 <= slice_idx < self.max_slices:
            self.current_slice = slice_idx
            self.update_display()
            self.slice_changed.emit(slice_idx)
    
    def set_window_level(self, center: int, width: int):
        """Устанавливает Window/Level"""
        self.window_center = center
        self.window_width = width
        self.update_display()
    
    def update_display(self):
        """Обновляет отображение"""
        slice_data = self.get_current_slice_data()
        
        if slice_data is None:
            return
        
        # Применяем Window/Level
        img_normalized = self._apply_window_level(slice_data)
        
        # Конвертируем в QPixmap
        height, width = img_normalized.shape
        bytes_per_line = width
        q_img = QImage(img_normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_img)
        
        # Масштабируем под размер виджета
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)
    
    def _apply_window_level(self, data: np.ndarray) -> np.ndarray:
        """Применяет Window/Level к данным"""
        min_val = self.window_center - self.window_width / 2
        max_val = self.window_center + self.window_width / 2
        
        # Нормализация
        windowed = np.clip(data, min_val, max_val)
        normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return normalized
    
    def wheelEvent(self, event):
        """Прокрутка колесом мыши для изменения среза"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.set_slice(self.current_slice + 1)
        else:
            self.set_slice(self.current_slice - 1)
    
    def resizeEvent(self, event):
        """Обновляем отображение при изменении размера"""
        super().resizeEvent(event)
        if self.pixmap():
            self.update_display()


class ProjectionManager(QWidget):
    """
    Менеджер окон проекций.
    Динамическая компоновка - легко добавлять/удалять проекции.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dicom_loader = None
        
        self.projections = {}  # {orientation: ProjectionView}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Настройка UI"""
        layout = QGridLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Создаем 3 стандартные проекции
        self._create_projection('axial', 0, 0)
        self._create_projection('sagittal', 0, 1)
        self._create_projection('coronal', 1, 0)
        
        # Резервное место для 4-й проекции (например, 3D)
        placeholder = QLabel("Резерв для\n3D / 4-й проекции")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                color: #666;
                border: 2px dashed #444;
            }
        """)
        layout.addWidget(placeholder, 1, 1)
    
    def _create_projection(self, orientation: str, row: int, col: int):
        """Создает виджет проекции"""
        projection = ProjectionView(orientation, self)
        self.projections[orientation] = projection
        self.layout().addWidget(projection, row, col)
    
    def set_dicom_loader(self, loader):
        """Устанавливает загрузчик DICOM"""
        self.dicom_loader = loader
    
    def update_views(self):
        """Обновляет все проекции с новыми данными"""
        if self.dicom_loader is None or self.dicom_loader.get_volume() is None:
            return
        
        volume = self.dicom_loader.get_volume()
        
        for projection in self.projections.values():
            projection.set_data(volume)
    
    def set_window_level(self, center: int, width: int):
        """Устанавливает Window/Level для всех проекций"""
        for projection in self.projections.values():
            projection.set_window_level(center, width)
    
    def add_projection(self, orientation: str, row: int, col: int):
        """
        Добавляет новую проекцию (для будущего расширения)
        Например, для 3D-вида или специальных проекций
        """
        if orientation not in self.projections:
            self._create_projection(orientation, row, col)
    
    def remove_projection(self, orientation: str):
        """Удаляет проекцию"""
        if orientation in self.projections:
            projection = self.projections.pop(orientation)
            self.layout().removeWidget(projection)
            projection.deleteLater()
