"""
Асинхронные worker'ы для выполнения длительных операций без блокировки UI.

Использует QThread для выполнения сегментации и постобработки в фоновом режиме.
"""

import time
from typing import Dict, Any, Tuple
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.processing import refine_mask


class SegmentationWorker(QThread):
    """
    Worker для асинхронной сегментации.
    
    Signals:
        finished: Испускается при успешном завершении (mask, statistics)
        error: Испускается при ошибке (error_message)
        progress: Испускается для обновления прогресса (percentage, message)
        log: Испускается для логирования (message)
    """
    
    finished = pyqtSignal(np.ndarray, dict)  # mask, stats
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # percentage, message
    log = pyqtSignal(str)
    
    def __init__(self,
                 model_instance: object,
                 volume: np.ndarray,
                 spacing: Tuple[float, float, float],
                 origin: Tuple[float, float, float],
                 direction: Tuple[float, ...],
                 roi_coords: Tuple[int, int, int, int, int, int],
                 **kwargs):
        """
        Инициализирует worker сегментации.
        
        Args:
            model_instance: Экземпляр модели сегментации (BaseSegmenter)
            volume: 3D массив изображения
            spacing: Размер вокселя
            origin: Координаты начала
            direction: Матрица направления
            roi_coords: Координаты ROI (z0, z1, y0, y1, x0, x1)
            **kwargs: Дополнительные параметры для модели
        """
        super().__init__()
        self.model = model_instance
        self.volume = volume
        self.spacing = spacing
        self.origin = origin
        self.direction = direction
        self.roi_coords = roi_coords
        self.model_params = kwargs
    
    def run(self) -> None:
        """Выполняет сегментацию в отдельном потоке."""
        try:
            start_time = time.time()
            
            self.log.emit(f"[INFO] Запуск сегментации: {self.model.display_name}")
            self.log.emit(f"[INFO] Устройство: {self.model.device.upper()}")
            self.log.emit(f"[INFO] ROI: z=[{self.roi_coords[0]}:{self.roi_coords[1]}]")
            
            self.progress.emit(5, "Инициализация модели...")
            
            # Выполняем сегментацию с callback для прогресса
            mask, stats = self.model.segment(
                volume=self.volume,
                spacing=self.spacing,
                origin=self.origin,
                direction=self.direction,
                roi_coords=self.roi_coords,
                progress_callback=self._progress_callback,
                **self.model_params
            )
            
            elapsed = time.time() - start_time
            
            # Добавляем информацию о времени если её нет
            if 'elapsed_time' not in stats:
                stats['elapsed_time'] = elapsed
            
            stats['timestamp'] = datetime.now()
            stats['organ_key'] = self.model.organ_key
            
            self.log.emit(f"[SUCCESS] Сегментация завершена за {elapsed:.1f}с")
            self.log.emit(f"[SUCCESS] Вокселей: {stats.get('voxel_count', 0):,}")
            
            self.finished.emit(mask, stats)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.log.emit(f"[ERROR] {str(e)}")
            self.error.emit(error_msg)
    
    def _progress_callback(self, percentage: int, message: str) -> None:
        """Callback для обновления прогресса."""
        self.progress.emit(percentage, message)


class MaskRefinementWorker(QThread):
    """
    Worker для асинхронной постобработки маски.
    
    Signals:
        finished: Испускается при успешном завершении (mask, statistics)
        error: Испускается при ошибке (error_message)
        progress: Испускается для обновления прогресса (percentage, message)
    """
    
    finished = pyqtSignal(np.ndarray, dict)  # mask, stats
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # percentage, message
    
    def __init__(self,
                 base_mask: np.ndarray,
                 volume: np.ndarray,
                 spacing: Tuple[float, float, float],
                 params: Dict[str, Any],
                 organ_key: str = "unknown"):
        """
        Инициализирует worker постобработки.
        
        Args:
            base_mask: Исходная маска для улучшения
            volume: 3D массив изображения (для HU фильтрации)
            spacing: Размер вокселя
            params: Параметры постобработки
            organ_key: Ключ органа для идентификации
        """
        super().__init__()
        self.base_mask = base_mask
        self.volume = volume
        self.spacing = spacing
        self.params = params
        self.organ_key = organ_key
    
    def run(self) -> None:
        """Выполняет постобработку в отдельном потоке."""
        try:
            start_time = time.time()
            
            self.progress.emit(5, "Начало обработки...")
            
            # Используем функцию из core.processing
            # Добавляем callback для детального прогресса
            refined_mask, stats = self._refine_with_progress()
            
            elapsed = time.time() - start_time
            stats['elapsed_time'] = elapsed
            stats['timestamp'] = datetime.now()
            stats['organ_key'] = self.organ_key
            
            self.progress.emit(100, "Готово!")
            self.finished.emit(refined_mask, stats)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def _refine_with_progress(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Выполняет постобработку с детальным прогрессом.
        
        Returns:
            Кортеж (refined_mask, statistics)
        """
        from scipy import ndimage
        
        mask = self.base_mask.copy()
        base_count = int(np.sum(mask))
        
        stats = {
            'base_count': base_count,
            'steps': []
        }
        
        # Шаг 1: Расширение по HU (40%)
        if self.params.get('dilation_iter', 0) > 0:
            self.progress.emit(20, f"Расширение по HU [{self.params['hu_min']}, {self.params['hu_max']}]...")
            
            lung_tissue = ((self.volume >= self.params['hu_min']) &
                          (self.volume <= self.params['hu_max'])).astype(np.uint8)
            dilated = ndimage.binary_dilation(mask, iterations=self.params['dilation_iter'])
            mask = dilated & lung_tissue
        
        # Шаг 2: Морфологическое закрытие (20%)
        if self.params.get('closing_size', 1) > 1:
            self.progress.emit(40, "Морфологическое закрытие...")
            structure = np.ones((self.params['closing_size'],
                               self.params['closing_size'],
                               self.params['closing_size']))
            mask = ndimage.binary_closing(mask, structure=structure)
        
        # Шаг 3: Удаление шума (20%)
        self.progress.emit(60, "Удаление мелких компонент...")
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            max_size = np.max(sizes)
            threshold = max_size * 0.01
            mask_cleaned = np.zeros_like(mask)
            for i, size in enumerate(sizes, 1):
                if size > threshold:
                    mask_cleaned[labeled == i] = 1
            mask = mask_cleaned
        
        # Шаг 4: Заполнение дыр (20%)
        if self.params.get('fill_holes', False):
            total_slices = mask.shape[0]
            for z in range(total_slices):
                if z % 10 == 0:
                    progress_pct = 80 + int((z / total_slices) * 15)
                    self.progress.emit(progress_pct, f"Заполнение дыр: {z}/{total_slices}")
                if np.any(mask[z]):
                    mask[z] = ndimage.binary_fill_holes(mask[z])
        
        self.progress.emit(95, "Финализация...")
        
        mask = mask.astype(np.uint8)
        final_count = int(np.sum(mask))
        improvement = ((final_count - base_count) / base_count * 100) if base_count > 0 else 0
        volume_ml = final_count * np.prod(self.spacing) / 1000.0
        
        stats.update({
            'final_count': final_count,
            'improvement_percent': improvement,
            'volume_ml': volume_ml,
            'params': self.params.copy()
        })
        
        return mask, stats


class DataLoadWorker(QThread):
    """
    Worker для асинхронной загрузки DICOM.
    
    Signals:
        finished: Испускается при успешной загрузке (volume, spacing, origin, direction, folder)
        error: Испускается при ошибке (error_message)
        progress: Испускается для обновления прогресса (percentage, message)
    """
    
    finished = pyqtSignal(np.ndarray, tuple, tuple, tuple, str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, folder_path: str):
        """
        Инициализирует worker загрузки.
        
        Args:
            folder_path: Путь к папке с DICOM файлами
        """
        super().__init__()
        self.folder_path = folder_path
    
    def run(self) -> None:
        """Выполняет загрузку в отдельном потоке."""
        try:
            from core.data_io import load_dicom_series
            
            self.progress.emit(10, "Сканирование DICOM файлов...")
            
            volume, spacing, origin, direction = load_dicom_series(self.folder_path)
            
            self.progress.emit(90, "Загрузка завершена")
            
            self.finished.emit(volume, spacing, origin, direction, self.folder_path)
            
        except Exception as e:
            self.error.emit(str(e))