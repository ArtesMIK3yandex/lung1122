"""
Базовый абстрактный класс для всех моделей сегментации.

Определяет интерфейс, который должны реализовывать все модели-плагины.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class BaseSegmenter(ABC):
    """
    Абстрактный базовый класс для моделей сегментации.
    
    Все новые модели должны наследовать этот класс и реализовывать
    все абстрактные методы.
    
    Attributes:
        display_name: Имя модели для отображения в UI
        organ_key: Ключ органа для словаря масок и цветов ('lung', 'liver', etc.)
    """
    
    display_name: str = "Base Segmenter"
    organ_key: str = "unknown"
    
    def __init__(self, use_cpu: bool = False, **kwargs):
        """
        Базовая инициализация модели.
        
        Args:
            use_cpu: Использовать CPU вместо GPU
            **kwargs: Дополнительные параметры для конкретной модели
        """
        self.use_cpu = use_cpu
        self.device = 'cpu' if use_cpu else self._auto_detect_device()
        self.model_params = kwargs
        self._model = None
    
    @abstractmethod
    def segment(self,
                volume: np.ndarray,
                spacing: Tuple[float, float, float],
                origin: Tuple[float, float, float],
                direction: Tuple[float, ...],
                roi_coords: Optional[Tuple[int, int, int, int, int, int]] = None,
                **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Выполняет сегментацию volume.
        
        Args:
            volume: 3D массив изображения (Z, Y, X)
            spacing: Размер вокселя (x, y, z) в мм
            origin: Координаты начала координат
            direction: Матрица направления
            roi_coords: Опциональные координаты ROI (z0, z1, y0, y1, x0, x1)
            **kwargs: Дополнительные параметры модели
            
        Returns:
            Кортеж (mask, statistics):
            - mask: Бинарная маска сегментации (uint8)
            - statistics: Словарь со статистикой работы модели
            
        Raises:
            NotImplementedError: Должен быть реализован в наследнике
        """
        raise NotImplementedError("Метод segment() должен быть реализован в наследнике")
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Загружает модель машинного обучения.
        
        Raises:
            NotImplementedError: Должен быть реализован в наследнике
        """
        raise NotImplementedError("Метод load_model() должен быть реализован в наследнике")
    
    def _auto_detect_device(self) -> str:
        """
        Автоматически определяет доступное устройство (cuda/cpu).
        
        Returns:
            Строка 'cuda' или 'cpu'
        """
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
    
    def validate_input(self,
                       volume: np.ndarray,
                       spacing: Tuple[float, float, float]) -> bool:
        """
        Проверяет корректность входных данных.
        
        Args:
            volume: 3D массив для проверки
            spacing: Spacing для проверки
            
        Returns:
            True если данные корректны
            
        Raises:
            ValueError: Если данные некорректны
        """
        if not isinstance(volume, np.ndarray):
            raise ValueError("Volume должен быть numpy array")
        
        if volume.ndim != 3:
            raise ValueError(f"Volume должен быть 3D, получен {volume.ndim}D")
        
        if len(spacing) != 3:
            raise ValueError(f"Spacing должен иметь 3 элемента, получено {len(spacing)}")
        
        if any(s <= 0 for s in spacing):
            raise ValueError("Все значения spacing должны быть положительными")
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели.
        
        Returns:
            Словарь с информацией о модели
        """
        return {
            'display_name': self.display_name,
            'organ_key': self.organ_key,
            'device': self.device,
            'use_cpu': self.use_cpu,
            'class': self.__class__.__name__,
            'params': self.model_params
        }
    
    def preprocess_volume(self,
                          volume: np.ndarray,
                          roi_coords: Optional[Tuple[int, int, int, int, int, int]] = None
                          ) -> np.ndarray:
        """
        Препроцессинг volume перед сегментацией.
        
        Args:
            volume: Исходный volume
            roi_coords: Опциональные координаты ROI
            
        Returns:
            Препроцессированный volume
        """
        if roi_coords is not None:
            z0, z1, y0, y1, x0, x1 = roi_coords
            return volume[z0:z1+1, y0:y1+1, x0:x1+1].copy()
        return volume.copy()
    
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Постпроцессинг маски после сегментации.
        
        Args:
            mask: Исходная маска
            
        Returns:
            Обработанная маска
        """
        # Базовая реализация - просто конвертируем в uint8
        return (mask > 0).astype(np.uint8)
    
    def estimate_memory_usage(self, volume_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """
        Оценивает требуемую память для сегментации.
        
        Args:
            volume_shape: Размер volume
            
        Returns:
            Словарь с оценкой памяти (в MB)
        """
        voxels = np.prod(volume_shape)
        
        # Примерные оценки
        volume_mb = voxels * 2 / (1024 * 1024)  # int16
        mask_mb = voxels / (1024 * 1024)  # uint8
        
        return {
            'volume_mb': volume_mb,
            'mask_mb': mask_mb,
            'estimated_total_mb': volume_mb + mask_mb + 100  # +100 для модели
        }
    
    def supports_batch_processing(self) -> bool:
        """
        Указывает, поддерживает ли модель пакетную обработку.
        
        Returns:
            True если поддерживается
        """
        return False
    
    def __repr__(self) -> str:
        """Строковое представление модели."""
        return f"{self.__class__.__name__}(display_name='{self.display_name}', organ='{self.organ_key}', device='{self.device}')"