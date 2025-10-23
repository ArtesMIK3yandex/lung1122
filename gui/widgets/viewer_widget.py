"""
Базовый функционал просмотра - НЕОТКЛЮЧАЕМЫЙ модуль.
Включает: управление срезами, Window/Level, отображение HU, навигацию.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QPushButton, QSpinBox, QComboBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal


class ViewerWidget(QWidget):
    """Виджет базового функционала просмотра"""
    
    # Сигналы
    slice_changed = pyqtSignal(int)
    window_level_changed = pyqtSignal(int, int)  # center, width
    slice_range_changed = pyqtSignal(int, int)   # start, end
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dicom_loader = None
        self.logger = None
        
        self.start_slice = 0
        self.end_slice = 0
        self.max_slices = 0
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Настройка UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # === ГРУППА: Управление срезами ===
        slice_group = QGroupBox("Управление срезами")
        slice_layout = QVBoxLayout()
        
        # Текущий срез
        current_slice_layout = QHBoxLayout()
        current_slice_layout.addWidget(QLabel("Срез:"))
        self.slice_label = QLabel("0 / 0")
        current_slice_layout.addWidget(self.slice_label)
        current_slice_layout.addStretch()
        slice_layout.addLayout(current_slice_layout)
        
        # Слайдер среза
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self._on_slice_slider_changed)
        slice_layout.addWidget(self.slice_slider)
        
        slice_group.setLayout(slice_layout)
        layout.addWidget(slice_group)
        
        # === ГРУППА: Рабочая область срезов (КРИТИЧНО) ===
        work_range_group = QGroupBox("Рабочая область")
        work_range_layout = QVBoxLayout()
        
        # Начальный срез
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("От среза:"))
        self.start_slice_spinbox = QSpinBox()
        self.start_slice_spinbox.setMinimum(0)
        self.start_slice_spinbox.valueChanged.connect(self._on_range_changed)
        start_layout.addWidget(self.start_slice_spinbox)
        work_range_layout.addLayout(start_layout)
        
        # Конечный срез
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("До среза:"))
        self.end_slice_spinbox = QSpinBox()
        self.end_slice_spinbox.setMinimum(0)
        self.end_slice_spinbox.valueChanged.connect(self._on_range_changed)
        end_layout.addWidget(self.end_slice_spinbox)
        work_range_layout.addLayout(end_layout)
        
        # Кнопка сброса
        self.reset_range_btn = QPushButton("Сброс выбора")
        self.reset_range_btn.clicked.connect(self._on_reset_range)
        work_range_layout.addWidget(self.reset_range_btn)
        
        work_range_group.setLayout(work_range_layout)
        layout.addWidget(work_range_group)
        
        # === ГРУППА: Window/Level ===
        wl_group = QGroupBox("Window/Level")
        wl_layout = QVBoxLayout()
        
        # Center
        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center:"))
        self.wl_center_spinbox = QSpinBox()
        self.wl_center_spinbox.setMinimum(-1024)
        self.wl_center_spinbox.setMaximum(3071)
        self.wl_center_spinbox.setValue(40)
        self.wl_center_spinbox.valueChanged.connect(self._on_wl_changed)
        center_layout.addWidget(self.wl_center_spinbox)
        wl_layout.addLayout(center_layout)
        
        # Width
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        self.wl_width_spinbox = QSpinBox()
        self.wl_width_spinbox.setMinimum(1)
        self.wl_width_spinbox.setMaximum(4095)
        self.wl_width_spinbox.setValue(400)
        self.wl_width_spinbox.valueChanged.connect(self._on_wl_changed)
        width_layout.addWidget(self.wl_width_spinbox)
        wl_layout.addLayout(width_layout)
        
        # Предустановки W/L
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Предустановки:"))
        self.wl_preset_combo = QComboBox()
        self.wl_preset_combo.addItems([
            "Пользовательское",
            "Лёгкие (-600/1500)",
            "Медиастинум (40/400)",
            "Кости (300/1500)",
            "Мягкие ткани (40/350)"
        ])
        self.wl_preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.wl_preset_combo)
        wl_layout.addLayout(preset_layout)
        
        wl_group.setLayout(wl_layout)
        layout.addWidget(wl_group)
        
        # === ГРУППА: Информация ===
        info_group = QGroupBox("Информация")
        info_layout = QVBoxLayout()
        
        self.hu_label = QLabel("HU: --")
        info_layout.addWidget(self.hu_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
    
    def set_dicom_loader(self, loader):
        """Устанавливает загрузчик DICOM"""
        self.dicom_loader = loader
    
    def set_logger(self, logger):
        """Устанавливает логгер"""
        self.logger = logger
    
    def update_from_data(self):
        """Обновление элементов управления после загрузки данных"""
        if self.dicom_loader is None:
            return
        
        volume = self.dicom_loader.get_volume()
        if volume is None:
            return
        
        self.max_slices = volume.shape[0]
        
        # Обновляем слайдер
        self.slice_slider.setMaximum(self.max_slices - 1)
        self.slice_slider.setValue(self.max_slices // 2)
        
        # Обновляем spinbox'ы диапазона
        self.start_slice_spinbox.setMaximum(self.max_slices - 1)
        self.end_slice_spinbox.setMaximum(self.max_slices - 1)
        
        self.start_slice = 0
        self.end_slice = self.max_slices - 1
        
        self.start_slice_spinbox.setValue(self.start_slice)
        self.end_slice_spinbox.setValue(self.end_slice)
        
        self._update_slice_label()
    
    def _on_slice_slider_changed(self, value):
        """Обработка изменения среза"""
        self._update_slice_label()
        self.slice_changed.emit(value)
        
        if self.logger:
            self.logger.log_slice_change(value)
    
    def _update_slice_label(self):
        """Обновление метки текущего среза"""
        current = self.slice_slider.value()
        total = self.slice_slider.maximum() + 1
        self.slice_label.setText(f"{current} / {total}")
    
    def _on_range_changed(self):
        """Обработка изменения рабочего диапазона"""
        self.start_slice = self.start_slice_spinbox.value()
        self.end_slice = self.end_slice_spinbox.value()
        
        # Валидация
        if self.start_slice > self.end_slice:
            self.end_slice_spinbox.setValue(self.start_slice)
            self.end_slice = self.start_slice
        
        self.slice_range_changed.emit(self.start_slice, self.end_slice)
        
        if self.logger:
            self.logger.log_slice_range_set(self.start_slice, self.end_slice)
    
    def _on_reset_range(self):
        """Сброс рабочего диапазона"""
        self.start_slice_spinbox.setValue(0)
        self.end_slice_spinbox.setValue(self.max_slices - 1)
    
    def _on_wl_changed(self):
        """Обработка изменения Window/Level"""
        center = self.wl_center_spinbox.value()
        width = self.wl_width_spinbox.value()
        
        self.wl_preset_combo.setCurrentIndex(0)  # "Пользовательское"
        self.window_level_changed.emit(center, width)
    
    def _on_preset_changed(self, index):
        """Обработка выбора предустановки W/L"""
        presets = {
            1: (-600, 1500),  # Лёгкие
            2: (40, 400),     # Медиастинум
            3: (300, 1500),   # Кости
            4: (40, 350)      # Мягкие ткани
        }
        
        if index in presets:
            center, width = presets[index]
            self.wl_center_spinbox.setValue(center)
            self.wl_width_spinbox.setValue(width)
    
    def update_hu_value(self, hu_value: float):
        """Обновление отображаемого значения HU"""
        self.hu_label.setText(f"HU: {hu_value:.1f}")
    
    def get_slice_range(self) -> tuple:
        """Возвращает выбранный диапазон срезов"""
        return (self.start_slice, self.end_slice)