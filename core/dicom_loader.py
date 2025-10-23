"""
Загрузчик DICOM-файлов с поддержкой:
- Рекурсивного сканирования папок
- Группировки по сериям (Series Instance UID)
- Обработки ошибок
"""

import pydicom
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np


class DICOMSeries:
    """Представление серии DICOM-снимков"""
    
    def __init__(self, series_uid: str, series_description: str = ""):
        self.series_uid = series_uid
        self.series_description = series_description
        self.files: List[Path] = []
        self.slices: List[pydicom.Dataset] = []
    
    def __str__(self):
        desc = self.series_description or "Без описания"
        return f"{desc} ({len(self.files)} файлов)"


class DICOMLoader:
    """Загрузчик и менеджер DICOM-данных"""
    
    def __init__(self):
        self.series_dict: Dict[str, DICOMSeries] = {}
        self.current_series: Optional[DICOMSeries] = None
        self.volume_data: Optional[np.ndarray] = None
        self.pixel_spacing = None
        self.slice_thickness = None
    
    def scan_directory(self, path: Path, recursive: bool = True) -> Dict[str, DICOMSeries]:
        """
        Сканирует директорию на наличие DICOM-файлов
        
        Args:
            path: Путь к директории
            recursive: Рекурсивный поиск в подпапках
        
        Returns:
            Словарь серий {series_uid: DICOMSeries}
        """
        self.series_dict.clear()
        dicom_files = self._find_dicom_files(path, recursive)
        
        print(f"Найдено {len(dicom_files)} DICOM-файлов")
        
        # Группировка по сериям
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                series_uid = getattr(ds, 'SeriesInstanceUID', 'unknown')
                
                if series_uid not in self.series_dict:
                    series_desc = getattr(ds, 'SeriesDescription', '')
                    self.series_dict[series_uid] = DICOMSeries(series_uid, series_desc)
                
                self.series_dict[series_uid].files.append(file_path)
                
            except Exception as e:
                print(f"⚠️ Ошибка чтения {file_path}: {e}")
                continue
        
        return self.series_dict
    
    def _find_dicom_files(self, path: Path, recursive: bool) -> List[Path]:
        """Находит все DICOM-файлы в директории"""
        dicom_files = []
        
        if path.is_file():
            # Если передан файл, проверяем только его
            if self._is_dicom_file(path):
                dicom_files.append(path)
        else:
            # Сканируем директорию
            pattern = "**/*" if recursive else "*"
            for file_path in path.glob(pattern):
                if file_path.is_file() and self._is_dicom_file(file_path):
                    dicom_files.append(file_path)
        
        return dicom_files
    
    def _is_dicom_file(self, file_path: Path) -> bool:
        """Проверяет, является ли файл DICOM"""
        # Проверка по расширению
        if file_path.suffix.lower() in ['.dcm', '.dicom']:
            return True
        
        # Проверка по магическим байтам DICM
        try:
            with open(file_path, 'rb') as f:
                f.seek(128)
                return f.read(4) == b'DICM'
        except:
            return False
    
    def load_series(self, series_uid: str) -> bool:
        """
        Загружает серию в память и строит 3D-объем
        
        Args:
            series_uid: UID серии для загрузки
        
        Returns:
            True при успешной загрузке
        """
        if series_uid not in self.series_dict:
            print(f"⚠️ Серия {series_uid} не найдена")
            return False
        
        series = self.series_dict[series_uid]
        print(f"Загрузка серии: {series}")
        
        # Загрузка всех срезов
        slices = []
        for file_path in series.files:
            try:
                ds = pydicom.dcmread(str(file_path))
                slices.append(ds)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {file_path}: {e}")
                continue
        
        if not slices:
            print("⚠️ Не удалось загрузить ни одного среза")
            return False
        
        # Сортировка срезов по позиции
        slices.sort(key=lambda x: float(getattr(x, 'ImagePositionPatient', [0, 0, 0])[2]))
        
        # Построение 3D-объема
        try:
            self.volume_data = np.stack([s.pixel_array for s in slices])
            
            # Применение Rescale Slope/Intercept для получения HU
            if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                slope = float(slices[0].RescaleSlope)
                intercept = float(slices[0].RescaleIntercept)
                self.volume_data = self.volume_data * slope + intercept
            
            # Сохранение метаданных
            self.pixel_spacing = getattr(slices[0], 'PixelSpacing', [1.0, 1.0])
            self.slice_thickness = getattr(slices[0], 'SliceThickness', 1.0)
            
            series.slices = slices
            self.current_series = series
            
            print(f"✓ Загружен объем: {self.volume_data.shape}")
            return True
            
        except Exception as e:
            print(f"⚠️ Ошибка построения объема: {e}")
            return False
    
    def get_series_list(self) -> List[tuple]:
        """Возвращает список серий для отображения в UI"""
        return [(uid, str(series)) for uid, series in self.series_dict.items()]
    
    def get_volume(self) -> Optional[np.ndarray]:
        """Возвращает загруженный 3D-объем"""
        return self.volume_data
    
    def get_slice(self, index: int, orientation: str = 'axial') -> Optional[np.ndarray]:
        """
        Возвращает срез в заданной ориентации
        
        Args:
            index: Индекс среза
            orientation: 'axial', 'sagittal', 'coronal'
        """
        if self.volume_data is None:
            return None
        
        try:
            if orientation == 'axial':
                return self.volume_data[index, :, :]
            elif orientation == 'sagittal':
                return self.volume_data[:, :, index]
            elif orientation == 'coronal':
                return self.volume_data[:, index, :]
        except IndexError:
            return None
    
    def get_metadata(self) -> dict:
        """Возвращает метаданные текущей серии"""
        if self.current_series and self.current_series.slices:
            ds = self.current_series.slices[0]
            return {
                'patient_name': str(getattr(ds, 'PatientName', 'N/A')),
                'patient_id': str(getattr(ds, 'PatientID', 'N/A')),
                'study_date': str(getattr(ds, 'StudyDate', 'N/A')),
                'modality': str(getattr(ds, 'Modality', 'N/A')),
                'series_description': self.current_series.series_description,
                'num_slices': len(self.current_series.slices)
            }
        return {}
