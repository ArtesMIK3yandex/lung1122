"""
Модуль для постобработки масок сегментации.

Содержит алгоритмы морфологической обработки, фильтрации и улучшения масок.
Полностью независим от GUI.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import ndimage


def refine_mask(base_mask: np.ndarray,
                volume: np.ndarray,
                spacing: Tuple[float, float, float],
                params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Улучшает маску сегментации с использованием морфологических операций.
    
    Args:
        base_mask: Исходная бинарная маска
        volume: Исходный 3D volume (для HU фильтрации)
        spacing: Размер вокселя (x, y, z)
        params: Словарь параметров:
            - hu_min: Минимальное HU значение
            - hu_max: Максимальное HU значение
            - dilation_iter: Количество итераций расширения
            - closing_size: Размер структурного элемента для closing
            - fill_holes: Заполнять ли дыры
            
    Returns:
        Кортеж (refined_mask, statistics):
        - refined_mask: Улучшенная маска
        - statistics: Словарь со статистикой обработки
        
    Example:
        >>> params = {'hu_min': -1000, 'hu_max': -300, 'dilation_iter': 2,
        ...           'closing_size': 3, 'fill_holes': True}
        >>> refined, stats = refine_mask(mask, volume, spacing, params)
    """
    mask = base_mask.copy()
    base_count = int(np.sum(mask))
    
    stats = {
        'base_count': base_count,
        'steps': []
    }
    
    # Шаг 1: Расширение по HU диапазону
    if params.get('dilation_iter', 0) > 0:
        mask, step_stats = _apply_hu_dilation(
            mask, volume, params['hu_min'], params['hu_max'], params['dilation_iter']
        )
        stats['steps'].append({'name': 'HU Dilation', **step_stats})
    
    # Шаг 2: Морфологическое закрытие
    if params.get('closing_size', 1) > 1:
        mask, step_stats = _apply_morphological_closing(mask, params['closing_size'])
        stats['steps'].append({'name': 'Morphological Closing', **step_stats})
    
    # Шаг 3: Удаление мелких компонент
    mask, step_stats = _remove_small_components(mask)
    stats['steps'].append({'name': 'Noise Removal', **step_stats})
    
    # Шаг 4: Заполнение дыр
    if params.get('fill_holes', False):
        mask, step_stats = _fill_holes_2d(mask)
        stats['steps'].append({'name': 'Hole Filling', **step_stats})
    
    # Финальная статистика
    final_count = int(np.sum(mask))
    improvement = ((final_count - base_count) / base_count * 100) if base_count > 0 else 0
    volume_ml = final_count * np.prod(spacing) / 1000.0
    
    stats.update({
        'final_count': final_count,
        'improvement_percent': improvement,
        'volume_ml': volume_ml,
        'params': params.copy()
    })
    
    return mask.astype(np.uint8), stats


def _apply_hu_dilation(mask: np.ndarray,
                       volume: np.ndarray,
                       hu_min: float,
                       hu_max: float,
                       iterations: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Расширяет маску в пределах заданного HU диапазона.
    
    Args:
        mask: Исходная маска
        volume: 3D volume
        hu_min: Минимальное HU
        hu_max: Максимальное HU
        iterations: Количество итераций расширения
        
    Returns:
        Кортеж (mask, stats)
    """
    count_before = int(np.sum(mask))
    
    # Создаем маску легочной ткани по HU
    lung_tissue = ((volume >= hu_min) & (volume <= hu_max)).astype(np.uint8)
    
    # Расширяем исходную маску
    dilated = ndimage.binary_dilation(mask, iterations=iterations)
    
    # Пересекаем с легочной тканью
    mask = dilated & lung_tissue
    
    count_after = int(np.sum(mask))
    
    return mask, {
        'voxels_before': count_before,
        'voxels_after': count_after,
        'voxels_added': count_after - count_before
    }


def _apply_morphological_closing(mask: np.ndarray, size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Применяет морфологическое закрытие для заполнения небольших пробелов.
    
    Args:
        mask: Исходная маска
        size: Размер структурного элемента
        
    Returns:
        Кортеж (mask, stats)
    """
    count_before = int(np.sum(mask))
    
    structure = np.ones((size, size, size))
    mask = ndimage.binary_closing(mask, structure=structure)
    
    count_after = int(np.sum(mask))
    
    return mask, {
        'voxels_before': count_before,
        'voxels_after': count_after,
        'voxels_added': count_after - count_before
    }


def _remove_small_components(mask: np.ndarray, 
                             threshold_ratio: float = 0.01) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Удаляет мелкие связные компоненты (шум).
    
    Args:
        mask: Исходная маска
        threshold_ratio: Порог размера компоненты относительно максимальной
        
    Returns:
        Кортеж (mask, stats)
    """
    count_before = int(np.sum(mask))
    
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask, {
            'voxels_before': count_before,
            'voxels_after': 0,
            'components_removed': 0
        }
    
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    max_size = np.max(sizes)
    threshold = max_size * threshold_ratio
    
    mask_cleaned = np.zeros_like(mask)
    components_kept = 0
    
    for i, size in enumerate(sizes, 1):
        if size > threshold:
            mask_cleaned[labeled == i] = 1
            components_kept += 1
    
    count_after = int(np.sum(mask_cleaned))
    
    return mask_cleaned, {
        'voxels_before': count_before,
        'voxels_after': count_after,
        'voxels_removed': count_before - count_after,
        'components_total': num_features,
        'components_kept': components_kept,
        'components_removed': num_features - components_kept
    }


def _fill_holes_2d(mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Заполняет дыры в маске (2D операция на каждом срезе).
    
    Args:
        mask: Исходная маска
        
    Returns:
        Кортеж (mask, stats)
    """
    count_before = int(np.sum(mask))
    total_slices = mask.shape[0]
    slices_processed = 0
    
    for z in range(total_slices):
        if np.any(mask[z]):
            mask[z] = ndimage.binary_fill_holes(mask[z])
            slices_processed += 1
    
    count_after = int(np.sum(mask))
    
    return mask, {
        'voxels_before': count_before,
        'voxels_after': count_after,
        'voxels_added': count_after - count_before,
        'slices_processed': slices_processed
    }


def calculate_mask_difference(mask1: np.ndarray, mask2: np.ndarray) -> Dict[str, Any]:
    """
    Вычисляет разницу между двумя масками.
    
    Args:
        mask1: Первая маска
        mask2: Вторая маска
        
    Returns:
        Словарь со статистикой различий
        
    Example:
        >>> diff = calculate_mask_difference(original_mask, refined_mask)
        >>> print(f"Added voxels: {diff['added_voxels']}")
    """
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    # Вычисляем различия
    added = mask2_bool & ~mask1_bool
    removed = mask1_bool & ~mask2_bool
    unchanged = mask1_bool & mask2_bool
    
    return {
        'mask1_count': int(np.sum(mask1_bool)),
        'mask2_count': int(np.sum(mask2_bool)),
        'added_voxels': int(np.sum(added)),
        'removed_voxels': int(np.sum(removed)),
        'unchanged_voxels': int(np.sum(unchanged)),
        'dice_coefficient': calculate_dice_coefficient(mask1_bool, mask2_bool)
    }


def calculate_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Вычисляет коэффициент Dice между двумя масками.
    
    Args:
        mask1: Первая маска
        mask2: Вторая маска
        
    Returns:
        Dice коэффициент (0.0 - 1.0)
        
    Example:
        >>> dice = calculate_dice_coefficient(mask1, mask2)
        >>> print(f"Dice: {dice:.3f}")
    """
    intersection = np.sum(mask1 & mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)
    
    if sum_masks == 0:
        return 1.0  # Обе маски пустые - считаем идеальным совпадением
    
    return 2.0 * intersection / sum_masks


def apply_preset(preset_name: str, 
                 presets_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Возвращает параметры для заданного пресета.
    
    Args:
        preset_name: Название пресета ('conservative', 'balanced', 'aggressive')
        presets_config: Словарь с конфигурацией пресетов
        
    Returns:
        Словарь с параметрами пресета
        
    Raises:
        KeyError: Если пресет не найден
        
    Example:
        >>> params = apply_preset('conservative', config['processing']['presets'])
    """
    if preset_name not in presets_config:
        raise KeyError(f"Пресет '{preset_name}' не найден. "
                      f"Доступные: {list(presets_config.keys())}")
    
    return presets_config[preset_name].copy()