"""
Реализация сегментации лёгких с использованием библиотеки lungmask.

Этот модуль является плагином и может быть заменен или дополнен
другими моделями без изменения основного кода.
"""

import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
import SimpleITK as sitk
import torch

from models.base_model import BaseSegmenter


class LungMaskSegmenter(BaseSegmenter):
    """
    Сегментация лёгких с использованием lungmask.
    
    Использует предобученную модель R231CovidWeb для сегментации
    лёгких на КТ изображениях.
    """
    
    display_name: str = "Сегментация лёгких (LungMask)"
    organ_key: str = "lung"
    
    def __init__(self, 
                 use_cpu: bool = False,
                 model_name: str = "R231CovidWeb",
                 batch_size: int = 20,
                 **kwargs):
        """
        Инициализирует модель lungmask.
        
        Args:
            use_cpu: Принудительно использовать CPU
            model_name: Название модели lungmask ('R231CovidWeb', 'LTRCLobes', etc.)
            batch_size: Размер батча для обработки больших volume
            **kwargs: Дополнительные параметры
        """
        super().__init__(use_cpu=use_cpu, **kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self._inferer = None
    
    def load_model(self) -> None:
        """
        Загружает модель lungmask.
        
        Raises:
            ImportError: Если lungmask не установлен
            RuntimeError: Если не удалось загрузить модель
        """
        try:
            from lungmask import LMInferer
            
            print(f"[INFO] Загрузка lungmask модели: {self.model_name}")
            self._inferer = LMInferer(
                modelname=self.model_name,
                force_cpu=self.use_cpu
            )
            print(f"[INFO] Модель загружена на {self.device.upper()}")
            
        except ImportError:
            raise ImportError(
                "lungmask не установлен. Установите: pip install lungmask"
            )
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить модель lungmask: {e}")
    
    def segment(self,
                volume: np.ndarray,
                spacing: Tuple[float, float, float],
                origin: Tuple[float, float, float],
                direction: Tuple[float, ...],
                roi_coords: Optional[Tuple[int, int, int, int, int, int]] = None,
                progress_callback: Optional[callable] = None,
                **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Выполняет сегментацию лёгких.
        
        Args:
            volume: 3D массив КТ изображения
            spacing: Размер вокселя
            origin: Координаты начала
            direction: Матрица направления
            roi_coords: Опциональный ROI (z0, z1, y0, y1, x0, x1)
            progress_callback: Функция для отчета о прогрессе
            **kwargs: Дополнительные параметры
            
        Returns:
            Кортеж (mask, statistics)
        """
        start_time = time.time()
        
        # Валидация входных данных
        self.validate_input(volume, spacing)
        
        # Загрузка модели если еще не загружена
        if self._inferer is None:
            self.load_model()
        
        # Препроцессинг - извлечение ROI если задан
        if roi_coords is not None:
            z0, z1, y0, y1, x0, x1 = roi_coords
            roi_volume = volume[z0:z1+1, y0:y1+1, x0:x1+1].copy()
            process_volume = roi_volume
        else:
            z0 = y0 = x0 = 0
            z1, y1, x1 = volume.shape[0]-1, volume.shape[1]-1, volume.shape[2]-1
            process_volume = volume.copy()
        
        if progress_callback:
            progress_callback(10, f"Подготовка данных: {process_volume.shape}")
        
        # Создание SimpleITK изображения
        roi_sitk = sitk.GetImageFromArray(process_volume)
        roi_sitk.SetSpacing(spacing)
        
        # Вычисление origin для ROI
        if roi_coords is not None:
            direction_matrix = np.array(direction).reshape(3, 3)
            roi_origin = (np.array(origin) + 
                         direction_matrix @ (np.array([x0, y0, z0]) * np.array(spacing)))
            roi_sitk.SetOrigin(roi_origin.tolist())
        else:
            roi_sitk.SetOrigin(origin)
        
        roi_sitk.SetDirection(direction)
        
        if progress_callback:
            progress_callback(20, f"Запуск lungmask на {self.device.upper()}")
        
        # Сегментация с обработкой OOM ошибок
        try:
            segmentation = self._run_segmentation(
                roi_sitk, 
                process_volume.shape[0],
                progress_callback
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and not self.use_cpu:
                if progress_callback:
                    progress_callback(30, "CUDA OOM! Переход на CPU...")
                
                # Очистка GPU памяти
                torch.cuda.empty_cache()
                
                # Перезагрузка на CPU
                self.use_cpu = True
                self.device = 'cpu'
                self.load_model()
                
                # Повторная попытка
                segmentation = self._run_segmentation(
                    roi_sitk,
                    process_volume.shape[0],
                    progress_callback
                )
            else:
                raise
        
        # Постпроцессинг - конвертация в бинарную маску
        mask_roi = self.postprocess_mask(segmentation)
        
        if progress_callback:
            progress_callback(90, "Финализация маски")
        
        # Создание полноразмерной маски если использовался ROI
        if roi_coords is not None:
            full_mask = np.zeros_like(volume, dtype=np.uint8)
            full_mask[z0:z1+1, y0:y1+1, x0:x1+1] = mask_roi
        else:
            full_mask = mask_roi
        
        elapsed_time = time.time() - start_time
        
        # Статистика
        voxel_count = int(np.sum(full_mask))
        statistics = {
            'voxel_count': voxel_count,
            'elapsed_time': elapsed_time,
            'device': self.device,
            'model_name': self.model_name,
            'roi_shape': process_volume.shape,
            'roi_coords': roi_coords,
            'used_batching': process_volume.shape[0] > 100
        }
        
        if progress_callback:
            progress_callback(100, f"Готово! {voxel_count:,} вокселей")
        
        return full_mask, statistics
    
    def _run_segmentation(self,
                         sitk_image: sitk.Image,
                         n_slices: int,
                         progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Выполняет сегментацию с опциональной пакетной обработкой.
        
        Args:
            sitk_image: SimpleITK изображение
            n_slices: Количество срезов
            progress_callback: Callback для прогресса
            
        Returns:
            Массив сегментации
        """
        # Для больших volume используем батч-обработку
        if n_slices > 100:
            if progress_callback:
                progress_callback(25, f"Пакетная обработка: {n_slices} срезов")
            return self._batch_segment(sitk_image, progress_callback)
        else:
            if progress_callback:
                progress_callback(25, f"Сегментация {n_slices} срезов")
            return self._inferer.apply(sitk_image)
    
    def _batch_segment(self,
                       roi_sitk: sitk.Image,
                       progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Сегментация по батчам для больших volume.
        
        Args:
            roi_sitk: SimpleITK изображение
            progress_callback: Callback для прогресса
            
        Returns:
            Полная маска сегментации
        """
        roi_array = sitk.GetArrayFromImage(roi_sitk)
        n_slices = roi_array.shape[0]
        result = np.zeros_like(roi_array, dtype=np.uint8)
        
        orig_spacing = roi_sitk.GetSpacing()
        orig_origin = roi_sitk.GetOrigin()
        orig_direction = roi_sitk.GetDirection()
        
        n_batches = (n_slices + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            i = batch_idx * self.batch_size
            end_i = min(i + self.batch_size, n_slices)
            
            if progress_callback:
                progress = 30 + int((batch_idx / n_batches) * 60)
                progress_callback(progress, f"Батч {batch_idx+1}/{n_batches}: срезы {i}-{end_i}")
            
            # Извлекаем батч
            batch = roi_array[i:end_i]
            batch_sitk = sitk.GetImageFromArray(batch)
            batch_sitk.SetSpacing(orig_spacing)
            batch_sitk.SetDirection(orig_direction)
            
            # Корректируем origin для батча
            batch_origin = list(orig_origin)
            batch_origin[2] = orig_origin[2] + i * orig_spacing[2]
            batch_sitk.SetOrigin(batch_origin)
            
            # Сегментация батча
            batch_seg = self._inferer.apply(batch_sitk)
            result[i:end_i] = batch_seg
        
        return result
    
    def supports_batch_processing(self) -> bool:
        """LungMask поддерживает пакетную обработку."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Расширенная информация о модели."""
        info = super().get_model_info()
        info.update({
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'supports_batching': True
        })
        return info