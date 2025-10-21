"""
Модуль для загрузки и сохранения медицинских данных.

Этот модуль полностью независим от GUI и содержит только логику работы с данными.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

import numpy as np
import SimpleITK as sitk
import nibabel as nib


def load_dicom_series(folder_path: str) -> Tuple[np.ndarray, Tuple[float, float, float], 
                                                  Tuple[float, float, float], Tuple[float, ...]]:
    """
    Загружает DICOM серию из указанной папки.
    
    Args:
        folder_path: Путь к папке с DICOM файлами
        
    Returns:
        Кортеж из (volume, spacing, origin, direction):
        - volume: 3D массив numpy с данными изображения
        - spacing: Размер вокселя (x, y, z) в мм
        - origin: Координаты начала координат (x, y, z)
        - direction: Матрица направления (9 элементов)
        
    Raises:
        ValueError: Если DICOM файлы не найдены
        RuntimeError: Если не удалось загрузить серию
        
    Example:
        >>> volume, spacing, origin, direction = load_dicom_series("/path/to/dicom")
        >>> print(f"Loaded volume shape: {volume.shape}")
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Путь не существует: {folder_path}")
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    
    if not dicom_names:
        raise ValueError(f"DICOM файлы не найдены в папке: {folder_path}")
    
    reader.SetFileNames(dicom_names)
    
    try:
        image = reader.Execute()
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить DICOM серию: {str(e)}")
    
    volume = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    
    return volume, spacing, origin, direction


def save_mask_nifti(mask: np.ndarray,
                    filepath: str,
                    spacing: Tuple[float, float, float],
                    origin: Tuple[float, float, float],
                    direction: Tuple[float, ...],
                    use_compression: bool = False,
                    prefer_sitk: bool = True) -> bool:
    """
    Сохраняет маску в формате NIfTI.
    
    Args:
        mask: 3D массив маски (uint8)
        filepath: Путь для сохранения файла
        spacing: Размер вокселя (x, y, z)
        origin: Координаты начала координат
        direction: Матрица направления
        use_compression: Использовать сжатие (.nii.gz)
        prefer_sitk: Предпочитать SimpleITK (если False, использовать nibabel)
        
    Returns:
        True если сохранение успешно, False иначе
        
    Example:
        >>> success = save_mask_nifti(mask, "output.nii", spacing, origin, direction)
    """
    # Убедимся, что расширение правильное
    if not (filepath.endswith('.nii') or filepath.endswith('.nii.gz')):
        filepath += '.nii.gz' if use_compression else '.nii'
    
    try:
        if prefer_sitk:
            _save_with_sitk(mask, filepath, spacing, origin, direction, use_compression)
        else:
            _save_with_nibabel(mask, filepath, spacing, origin, direction)
        return True
        
    except Exception as sitk_error:
        print(f"[WARNING] SimpleITK ошибка: {sitk_error}")
        
        # Fallback на nibabel
        try:
            _save_with_nibabel(mask, filepath, spacing, origin, direction)
            return True
        except Exception as nib_error:
            print(f"[ERROR] Nibabel ошибка: {nib_error}")
            return False


def _save_with_sitk(mask: np.ndarray,
                    filepath: str,
                    spacing: Tuple[float, float, float],
                    origin: Tuple[float, float, float],
                    direction: Tuple[float, ...],
                    use_compression: bool) -> None:
    """Сохранение через SimpleITK."""
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_sitk.SetSpacing(spacing)
    mask_sitk.SetOrigin(origin)
    mask_sitk.SetDirection(direction)
    sitk.WriteImage(mask_sitk, filepath, useCompression=use_compression)


def _save_with_nibabel(mask: np.ndarray,
                       filepath: str,
                       spacing: Tuple[float, float, float],
                       origin: Tuple[float, float, float],
                       direction: Tuple[float, ...]) -> None:
    """Сохранение через nibabel."""
    spacing_arr = np.array(spacing)
    direction_arr = np.array(direction).reshape(3, 3)
    origin_arr = np.array(origin)
    
    # Создаем affine матрицу
    affine = np.eye(4)
    for i in range(3):
        for j in range(3):
            affine[i, j] = direction_arr[i, j] * spacing_arr[j]
    affine[:3, 3] = origin_arr
    
    # nibabel требует транспонирования
    mask_transposed = mask.transpose(2, 1, 0)
    nii_img = nib.Nifti1Image(mask_transposed.astype(np.uint8), affine)
    nib.save(nii_img, filepath)


def export_history_to_file(history: list, filepath: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Экспортирует историю операций в текстовый файл.
    
    Args:
        history: Список записей истории
        filepath: Путь к файлу для сохранения
        config: Опциональный словарь конфигурации
        
    Returns:
        True если экспорт успешен
        
    Example:
        >>> history = [{"timestamp": datetime.now(), "message": "Test"}]
        >>> export_history_to_file(history, "history.txt")
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("LUNG SEGMENTER - ИСТОРИЯ ОПЕРАЦИЙ\n")
            f.write("=" * 70 + "\n")
            f.write(f"Экспортировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            for entry in history:
                if isinstance(entry, dict):
                    timestamp = entry.get('timestamp', 'N/A')
                    if isinstance(timestamp, datetime):
                        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    
                    f.write(f"[{timestamp}]\n")
                    f.write(f"Тип: {entry.get('type', 'unknown')}\n")
                    
                    if 'stats' in entry:
                        f.write("Статистика:\n")
                        for key, value in entry['stats'].items():
                            f.write(f"  - {key}: {value}\n")
                    
                    f.write("\n")
                else:
                    f.write(f"{entry}\n")
            
            f.write("=" * 70 + "\n")
            f.write("КОНЕЦ ИСТОРИИ\n")
            f.write("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Не удалось экспортировать историю: {e}")
        return False


def get_volume_statistics(volume: np.ndarray) -> Dict[str, Any]:
    """
    Вычисляет базовую статистику для volume.
    
    Args:
        volume: 3D массив данных
        
    Returns:
        Словарь со статистикой
        
    Example:
        >>> stats = get_volume_statistics(volume)
        >>> print(f"Min HU: {stats['min']}, Max HU: {stats['max']}")
    """
    return {
        'shape': volume.shape,
        'min': float(np.min(volume)),
        'max': float(np.max(volume)),
        'mean': float(np.mean(volume)),
        'std': float(np.std(volume)),
        'percentile_2': float(np.percentile(volume, 2)),
        'percentile_98': float(np.percentile(volume, 98))
    }


def get_mask_statistics(mask: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, Any]:
    """
    Вычисляет статистику для маски.
    
    Args:
        mask: 3D бинарная маска
        spacing: Размер вокселя
        
    Returns:
        Словарь со статистикой маски
        
    Example:
        >>> stats = get_mask_statistics(mask, spacing)
        >>> print(f"Volume: {stats['volume_ml']:.1f} ml")
    """
    voxel_count = int(np.sum(mask))
    voxel_volume_mm3 = np.prod(spacing)
    volume_ml = voxel_count * voxel_volume_mm3 / 1000.0
    
    # Вычисляем покрытие по срезам
    z_coverage = int(np.sum(np.sum(mask, axis=(1, 2)) > 0))
    
    return {
        'voxel_count': voxel_count,
        'volume_ml': volume_ml,
        'volume_cm3': volume_ml,
        'z_coverage': z_coverage,
        'z_total': mask.shape[0]
    }