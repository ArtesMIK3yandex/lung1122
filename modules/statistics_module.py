"""
Модуль статистики по исследованию.
Настраиваемый модуль (может быть скрыт администратором).
"""

from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QGroupBox, QTextEdit)
from modules.base_module import BaseModule
import numpy as np


class StatisticsModule(BaseModule):
    """Модуль отображения статистики по исследованию"""
    
    def get_module_info(self):
        """Метаданные модуля"""
        return {
            'id': 'statistics',
            'name': 'Статистика по исследованию',
            'version': '1.0.0',
            'description': 'Отображает статистическую информацию о загруженных данных',
            'removable': True  # Может быть скрыт администратором
        }
    
    def initialize(self):
        """Инициализация UI модуля"""
        layout = QVBoxLayout(self)
        
        # Группа статистики
        stats_group = QGroupBox("Статистика")
        stats_layout = QVBoxLayout()
        
        # Текстовое поле для вывода статистики
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setPlaceholderText("Загрузите данные для просмотра статистики")
        stats_layout.addWidget(self.stats_text)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        self._initialized = True
    
    def on_data_loaded(self, data_loader):
        """Обработка загрузки новых данных"""
        volume = data_loader.get_volume()
        metadata = data_loader.get_metadata()
        
        if volume is None:
            return
        
        # Вычисление статистики
        stats = self._calculate_statistics(volume, metadata)
        
        # Отображение статистики
        self._display_statistics(stats)
    
    def _calculate_statistics(self, volume: np.ndarray, metadata: dict) -> dict:
        """Вычисляет статистику по объему данных"""
        stats = {
            'shape': volume.shape,
            'min_hu': float(np.min(volume)),
            'max_hu': float(np.max(volume)),
            'mean_hu': float(np.mean(volume)),
            'std_hu': float(np.std(volume)),
            'median_hu': float(np.median(volume)),
            'total_voxels': volume.size,
            **metadata
        }
        
        return stats
    
    def _display_statistics(self, stats: dict):
        """Отображает статистику в текстовом поле"""
        text = f"""
<b>Информация о пациенте:</b><br>
Имя: {stats.get('patient_name', 'N/A')}<br>
ID: {stats.get('patient_id', 'N/A')}<br>
Дата исследования: {stats.get('study_date', 'N/A')}<br>
Модальность: {stats.get('modality', 'N/A')}<br>
<br>
<b>Параметры объема:</b><br>
Размерность: {stats['shape'][0]} × {stats['shape'][1]} × {stats['shape'][2]}<br>
Всего вокселей: {stats['total_voxels']:,}<br>
<br>
<b>Статистика HU:</b><br>
Минимум: {stats['min_hu']:.1f} HU<br>
Максимум: {stats['max_hu']:.1f} HU<br>
Среднее: {stats['mean_hu']:.1f} HU<br>
Медиана: {stats['median_hu']:.1f} HU<br>
Стд. отклонение: {stats['std_hu']:.1f} HU<br>
"""
        
        self.stats_text.setHtml(text)
        
        if self.logger:
            self.logger.log_action("view_statistics", stats_summary=stats)
