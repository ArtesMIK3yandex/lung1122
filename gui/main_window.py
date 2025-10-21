"""
Главное окно приложения Lung Segmenter GUI.

Модульная архитектура с разделением бизнес-логики и UI.
Поддержка упрощенного и расширенного режимов интерфейса.
"""

import os
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QSlider, QProgressBar, QMessageBox,
    QGroupBox, QRadioButton, QButtonGroup, QSpinBox, QCheckBox,
    QTextEdit, QComboBox
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QParallelAnimationGroup
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from core.state_manager import UIStateManager, UIState
from core.model_loader import ModelRegistry
from core.data_io import save_mask_nifti, export_history_to_file, get_mask_statistics
from gui.workers import SegmentationWorker, MaskRefinementWorker, DataLoadWorker
from gui.widgets.roi_selector import ROISelector, ROIManager


class LungSegmenterGUI(QMainWindow):
    """
    Главное окно приложения сегментации.
    
    Реализует модульную архитектуру с поддержкой:
    - Множественных моделей сегментации
    - Множественных масок для разных органов
    - Упрощенного и расширенного режимов UI
    - Централизованного управления состоянием
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Инициализирует главное окно.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        super().__init__()
        
        # Загрузка конфигурации
        self.config = self._load_config(config_path)
        
        # Данные приложения
        self.volume: Optional[np.ndarray] = None
        self.spacing: Optional[Tuple[float, float, float]] = None
        self.origin: Optional[Tuple[float, float, float]] = None
        self.direction: Optional[Tuple[float, ...]] = None
        self.dicom_folder: Optional[str] = None
        
        # Словарь масок для разных органов
        self.masks: Dict[str, np.ndarray] = {}  # {'lung': array, 'liver': array, ...}
        self.base_masks: Dict[str, np.ndarray] = {}  # Исходные маски до постобработки
        
        # Отображение
        self.current_slice: int = 0
        self.current_view: str = 'axial'
        self.window_center: int = self.config['display']['default_window_center']
        self.window_width: int = self.config['display']['default_window_width']
        self.auto_window: bool = self.config['display']['auto_window_enabled']
        self.saved_xlim: Optional[Tuple[float, float]] = None
        self.saved_ylim: Optional[Tuple[float, float]] = None
        
        # ROI менеджер
        self.roi_manager = ROIManager()
        self.roi_selector: Optional[ROISelector] = None
        
        # Модели сегментации
        self.model_registry = ModelRegistry()
        self.model_registry.load_models()
        self.current_model_name: Optional[str] = None
        
        # Состояние UI
        self.state_manager = UIStateManager()
        
        # История операций
        self.operation_history = []
        
        # Режим интерфейса
        self.simplified_mode: bool = self.config['ui']['modes']['simplified']
        
        # Инициализация UI
        self._init_ui()
        self._setup_state_manager()
        
        # Приветственное сообщение
        self._show_welcome_message()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загружает конфигурацию из YAML файла."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            QMessageBox.warning(
                None, "Конфигурация не найдена",
                f"Файл {config_path} не найден. Используются значения по умолчанию."
            )
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию."""
        return {
            'application': {
                'window_title': 'Lung Segmenter GUI',
                'window_size': [1600, 900]
            },
            'display': {
                'default_window_center': -600,
                'default_window_width': 1500,
                'auto_window_enabled': True,
                'presets': {
                    'lung': {'center': -600, 'width': 1500},
                    'mediastinum': {'center': 40, 'width': 400}
                }
            },
            'organ_colors': {
                'lung': 'lime',
                'liver': 'orange',
                'kidney': 'cyan',
                'default': 'red'
            },
            'ui': {
                'panels': {
                    'control_panel_width': 300,
                    'refine_panel_width': 350
                },
                'modes': {
                    'simplified': True
                },
                'animation': {
                    'duration_ms': 250,
                    'easing': 'InOutCubic'
                }
            },
            'processing': {
                'presets': {
                    'conservative': {
                        'hu_min': -950,
                        'hu_max': -350,
                        'dilation_iter': 1,
                        'closing_size': 3,
                        'fill_holes': True
                    },
                    'balanced': {
                        'hu_min': -1000,
                        'hu_max': -300,
                        'dilation_iter': 2,
                        'closing_size': 3,
                        'fill_holes': True
                    },
                    'aggressive': {
                        'hu_min': -1100,
                        'hu_max': -250,
                        'dilation_iter': 4,
                        'closing_size': 5,
                        'fill_holes': True
                    }
                },
                'parameter_ranges': {
                    'hu_min': {'min': -1200, 'max': -100, 'default': -1000},
                    'hu_max': {'min': -900, 'max': 100, 'default': -300},
                    'dilation_iter': {'min': 0, 'max': 10, 'default': 2},
                    'closing_size': {'min': 1, 'max': 10, 'default': 3}
                }
            }
        }
    
    def _init_ui(self) -> None:
        """Инициализирует пользовательский интерфейс."""
        # Настройка окна
        app_config = self.config['application']
        self.setWindowTitle(app_config['window_title'])
        size = app_config['window_size']
        self.setGeometry(100, 100, size[0], size[1])
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создание панелей
        self.left_panel = QWidget(central_widget)
        self.right_panel = QWidget(central_widget)
        self.refine_panel = QWidget(central_widget)
        
        # Кнопка переключения расширенной панели
        self.toggle_refine_button = QPushButton("▶", central_widget)
        self.toggle_refine_button.setCheckable(True)
        self.toggle_refine_button.setFixedSize(25, 50)
        self.toggle_refine_button.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14pt; }"
        )
        self.toggle_refine_button.toggled.connect(self._toggle_refinement_panel)
        
        # Заполнение панелей
        self._setup_left_panel()
        self._setup_right_panel()
        self._setup_refine_panel()
        
        # Начальное позиционирование
        self._update_panel_positions(animate=False)
    
    def _setup_left_panel(self) -> None:
        """Настраивает левую панель (просмотр изображений)."""
        layout = QVBoxLayout(self.left_panel)
        
        # Matplotlib canvas
        self.figure = Figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Слайдер срезов
        slice_layout = QHBoxLayout()
        self.slice_label = QLabel("Slice: 0/0")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        slice_layout.addWidget(self.slice_label)
        slice_layout.addWidget(self.slice_slider)
        layout.addLayout(slice_layout)
        
        # Progress bar
        progress_layout = QVBoxLayout()
        self.progress_message = QLabel("")
        self.progress_message.setAlignment(Qt.AlignCenter)
        self.progress_message.setStyleSheet("QLabel { color: #0066cc; font-weight: bold; }")
        progress_layout.addWidget(self.progress_message)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)
        
        # История операций
        history_group = QGroupBox("📋 История операций")
        history_layout = QVBoxLayout()
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(120)
        self.history_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                border: 1px solid #ccc;
            }
        """)
        history_layout.addWidget(self.history_text)
        
        # Кнопка экспорта истории
        self.btn_export_history = QPushButton("💾 Экспорт истории")
        self.btn_export_history.clicked.connect(self._export_history)
        history_layout.addWidget(self.btn_export_history)
        
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
    
    def _setup_right_panel(self) -> None:
        """Настраивает правую панель (управление)."""
        layout = QVBoxLayout(self.right_panel)
        
        # Группа загрузки
        load_group = QGroupBox("📁 Загрузка данных")
        load_layout = QVBoxLayout()
        self.btn_load = QPushButton("Загрузить DICOM")
        self.btn_load.clicked.connect(self._load_dicom)
        load_layout.addWidget(self.btn_load)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Группа отображения
        display_group = QGroupBox("🖼️ Отображение (HU)")
        display_layout = QVBoxLayout()
        
        self.cb_auto_window = QCheckBox("Авто Window/Level")
        self.cb_auto_window.setChecked(self.auto_window)
        self.cb_auto_window.toggled.connect(self._toggle_auto_window)
        display_layout.addWidget(self.cb_auto_window)
        
        # Window Center
        wc_layout = QHBoxLayout()
        wc_layout.addWidget(QLabel("Center:"))
        self.spin_wc = QSpinBox()
        self.spin_wc.setRange(-2000, 2000)
        self.spin_wc.setValue(self.window_center)
        self.spin_wc.setSingleStep(50)
        self.spin_wc.valueChanged.connect(self._on_window_changed)
        self.spin_wc.setEnabled(not self.auto_window)
        wc_layout.addWidget(self.spin_wc)
        display_layout.addLayout(wc_layout)
        
        # Window Width
        ww_layout = QHBoxLayout()
        ww_layout.addWidget(QLabel("Width:"))
        self.spin_ww = QSpinBox()
        self.spin_ww.setRange(1, 4000)
        self.spin_ww.setValue(self.window_width)
        self.spin_ww.setSingleStep(100)
        self.spin_ww.valueChanged.connect(self._on_window_changed)
        self.spin_ww.setEnabled(not self.auto_window)
        ww_layout.addWidget(self.spin_ww)
        display_layout.addLayout(ww_layout)
        
        # Пресеты
        preset_layout = QHBoxLayout()
        self.btn_lung_preset = QPushButton("Лёгкие")
        self.btn_lung_preset.clicked.connect(lambda: self._apply_window_preset('lung'))
        self.btn_mediastinum_preset = QPushButton("Медиастинум")
        self.btn_mediastinum_preset.clicked.connect(lambda: self._apply_window_preset('mediastinum'))
        preset_layout.addWidget(self.btn_lung_preset)
        preset_layout.addWidget(self.btn_mediastinum_preset)
        display_layout.addLayout(preset_layout)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Группа проекций
        view_group = QGroupBox("🔄 Проекция")
        view_layout = QHBoxLayout()
        self.view_group_buttons = QButtonGroup()
        
        self.rb_axial = QRadioButton("Axial (Z)")
        self.rb_coronal = QRadioButton("Coronal (Y)")
        self.rb_sagittal = QRadioButton("Sagittal (X)")
        self.rb_axial.setChecked(True)
        
        self.view_group_buttons.addButton(self.rb_axial)
        self.view_group_buttons.addButton(self.rb_coronal)
        self.view_group_buttons.addButton(self.rb_sagittal)
        
        self.rb_axial.toggled.connect(lambda: self._change_view('axial'))
        self.rb_coronal.toggled.connect(lambda: self._change_view('coronal'))
        self.rb_sagittal.toggled.connect(lambda: self._change_view('sagittal'))
        
        view_layout.addWidget(self.rb_axial)
        view_layout.addWidget(self.rb_coronal)
        view_layout.addWidget(self.rb_sagittal)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # Группа ROI
        roi_group = QGroupBox("📐 ROI")
        roi_layout = QVBoxLayout()
        
        roi_buttons_layout = QHBoxLayout()
        self.btn_draw_roi1 = QPushButton("ROI 1")
        self.btn_draw_roi1.clicked.connect(self._draw_roi1)
        self.btn_draw_roi1.setEnabled(False)
        self.btn_draw_roi1.setToolTip("Нарисуйте прямоугольник на начальном срезе")
        
        self.btn_draw_roi2 = QPushButton("ROI 2")
        self.btn_draw_roi2.clicked.connect(self._draw_roi2)
        self.btn_draw_roi2.setEnabled(False)
        self.btn_draw_roi2.setToolTip("Нарисуйте прямоугольник на конечном срезе")
        
        self.btn_reset_roi = QPushButton("↺")
        self.btn_reset_roi.clicked.connect(self._reset_roi)
        self.btn_reset_roi.setMaximumWidth(40)
        self.btn_reset_roi.setToolTip("Сбросить ROI")
        
        roi_buttons_layout.addWidget(self.btn_draw_roi1)
        roi_buttons_layout.addWidget(self.btn_draw_roi2)
        roi_buttons_layout.addWidget(self.btn_reset_roi)
        roi_layout.addLayout(roi_buttons_layout)
        
        self.roi_info_label = QLabel("ROI не заданы")
        self.roi_info_label.setWordWrap(True)
        self.roi_info_label.setStyleSheet("QLabel { font-size: 8pt; }")
        roi_layout.addWidget(self.roi_info_label)
        
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)
        
        # Группа сегментации
        seg_group = QGroupBox("🧠 Сегментация")
        seg_layout = QVBoxLayout()
        
        # Выбор модели
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_registry.list_models())
        if self.model_combo.count() > 0:
            self.current_model_name = self.model_combo.currentText()
        model_layout.addWidget(self.model_combo)
        seg_layout.addLayout(model_layout)
        
        self.btn_segment = QPushButton("▶️ Запустить сегментацию")
        self.btn_segment.clicked.connect(self._segment_roi)
        self.btn_segment.setEnabled(False)
        seg_layout.addWidget(self.btn_segment)
        
        self.cpu_checkbox = QCheckBox("Использовать CPU")
        self.cpu_checkbox.setToolTip("Если мало VRAM или ошибки CUDA")
        seg_layout.addWidget(self.cpu_checkbox)
        
        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)
        
        # Группа сохранения
        save_group = QGroupBox("💾 Сохранение")
        save_layout = QVBoxLayout()
        
        self.btn_save = QPushButton("💾 Сохранить маски")
        self.btn_save.clicked.connect(self._save_masks)
        self.btn_save.setEnabled(False)
        save_layout.addWidget(self.btn_save)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        layout.addStretch()
    
    def _setup_refine_panel(self) -> None:
        """Настраивает панель постобработки (выдвижную)."""
        self.refine_panel.setStyleSheet(
            "background-color: #f8f9fa; border-left: 1px solid #d3d3d3;"
        )
        
        layout = QVBoxLayout(self.refine_panel)
        
        refine_group = QGroupBox("🔧 Постобработка маски")
        refine_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout.addWidget(refine_group)
        
        refine_main_layout = QVBoxLayout(refine_group)
        
        # Выбор органа для обработки
        organ_layout = QHBoxLayout()
        organ_layout.addWidget(QLabel("Орган:"))
        self.organ_combo = QComboBox()
        organ_layout.addWidget(self.organ_combo)
        refine_main_layout.addLayout(organ_layout)
        
        # Пресеты (упрощенный режим)
        preset_group = QGroupBox("⚡ Быстрые настройки")
        preset_layout = QVBoxLayout()
        
        self.btn_conservative_preset = QPushButton("🛡️ Консервативный")
        self.btn_conservative_preset.clicked.connect(
            lambda: self._apply_processing_preset('conservative')
        )
        preset_layout.addWidget(self.btn_conservative_preset)
        
        self.btn_balanced_preset = QPushButton("⚖️ Сбалансированный")
        self.btn_balanced_preset.clicked.connect(
            lambda: self._apply_processing_preset('balanced')
        )
        preset_layout.addWidget(self.btn_balanced_preset)
        
        self.btn_aggressive_preset = QPushButton("🚀 Агрессивный")
        self.btn_aggressive_preset.clicked.connect(
            lambda: self._apply_processing_preset('aggressive')
        )
        preset_layout.addWidget(self.btn_aggressive_preset)
        
        preset_group.setLayout(preset_layout)
        refine_main_layout.addWidget(preset_group)
        
        # Кнопка переключения в расширенный режим
        self.btn_toggle_mode = QPushButton("🔧 Расширенные настройки...")
        self.btn_toggle_mode.clicked.connect(self._toggle_ui_mode)
        refine_main_layout.addWidget(self.btn_toggle_mode)
        
        # Детальные параметры (расширенный режим)
        self.advanced_params_widget = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_params_widget)
        
        # HU диапазон
        hu_group = QGroupBox("Диапазон HU")
        hu_layout = QVBoxLayout()
        
        params = self.config['processing']['parameter_ranges']
        
        hu_min_layout = QHBoxLayout()
        self.label_hu_min = QLabel(str(params['hu_min']['default']))
        self.slider_hu_min = QSlider(Qt.Horizontal)
        self.slider_hu_min.setRange(params['hu_min']['min'], params['hu_min']['max'])
        self.slider_hu_min.setValue(params['hu_min']['default'])
        hu_min_layout.addWidget(QLabel("Min:"))
        hu_min_layout.addWidget(self.slider_hu_min)
        hu_min_layout.addWidget(self.label_hu_min)
        hu_layout.addLayout(hu_min_layout)
        
        hu_max_layout = QHBoxLayout()
        self.label_hu_max = QLabel(str(params['hu_max']['default']))
        self.slider_hu_max = QSlider(Qt.Horizontal)
        self.slider_hu_max.setRange(params['hu_max']['min'], params['hu_max']['max'])
        self.slider_hu_max.setValue(params['hu_max']['default'])
        hu_max_layout.addWidget(QLabel("Max:"))
        hu_max_layout.addWidget(self.slider_hu_max)
        hu_max_layout.addWidget(self.label_hu_max)
        hu_layout.addLayout(hu_max_layout)
        
        self.slider_hu_min.valueChanged.connect(self._update_param_labels)
        self.slider_hu_max.valueChanged.connect(self._update_param_labels)
        
        hu_group.setLayout(hu_layout)
        advanced_layout.addWidget(hu_group)
        
        # Морфология
        morph_group = QGroupBox("Морфология")
        morph_layout = QVBoxLayout()
        
        dilation_layout = QHBoxLayout()
        self.label_dilation = QLabel(str(params['dilation_iter']['default']))
        self.slider_dilation = QSlider(Qt.Horizontal)
        self.slider_dilation.setRange(params['dilation_iter']['min'], params['dilation_iter']['max'])
        self.slider_dilation.setValue(params['dilation_iter']['default'])
        dilation_layout.addWidget(QLabel("Dilation:"))
        dilation_layout.addWidget(self.slider_dilation)
        dilation_layout.addWidget(self.label_dilation)
        morph_layout.addLayout(dilation_layout)
        
        closing_layout = QHBoxLayout()
        self.label_closing = QLabel(str(params['closing_size']['default']))
        self.slider_closing = QSlider(Qt.Horizontal)
        self.slider_closing.setRange(params['closing_size']['min'], params['closing_size']['max'])
        self.slider_closing.setValue(params['closing_size']['default'])
        closing_layout.addWidget(QLabel("Closing:"))
        closing_layout.addWidget(self.slider_closing)
        closing_layout.addWidget(self.label_closing)
        morph_layout.addLayout(closing_layout)
        
        self.slider_dilation.valueChanged.connect(self._update_param_labels)
        self.slider_closing.valueChanged.connect(self._update_param_labels)
        
        self.cb_fill_holes = QCheckBox("Заполнять дыры")
        self.cb_fill_holes.setChecked(True)
        morph_layout.addWidget(self.cb_fill_holes)
        
        morph_group.setLayout(morph_layout)
        advanced_layout.addWidget(morph_group)
        
        # Показываем/скрываем в зависимости от режима
        self.advanced_params_widget.setVisible(not self.simplified_mode)
        refine_main_layout.addWidget(self.advanced_params_widget)
        
        # Статистика
        stats_group = QGroupBox("📊 Статистика")
        stats_layout = QVBoxLayout()
        self.label_mask_stats = QLabel("Нет данных")
        self.label_mask_stats.setWordWrap(True)
        stats_layout.addWidget(self.label_mask_stats)
        stats_group.setLayout(stats_layout)
        refine_main_layout.addWidget(stats_group)
        
        # Кнопки управления
        control_layout = QHBoxLayout()
        self.btn_apply_refinement = QPushButton("✓ Применить")
        self.btn_apply_refinement.clicked.connect(self._apply_mask_refinement)
        self.btn_apply_refinement.setEnabled(False)
        
        self.btn_reset_mask = QPushButton("↺ Сброс")
        self.btn_reset_mask.clicked.connect(self._reset_to_base_mask)
        self.btn_reset_mask.setEnabled(False)
        
        control_layout.addWidget(self.btn_apply_refinement)
        control_layout.addWidget(self.btn_reset_mask)
        refine_main_layout.addLayout(control_layout)
    
    def _setup_state_manager(self) -> None:
        """Настраивает менеджер состояний UI."""
        widgets = {
            'btn_load': self.btn_load,
            'btn_draw_roi1': self.btn_draw_roi1,
            'btn_draw_roi2': self.btn_draw_roi2,
            'btn_reset_roi': self.btn_reset_roi,
            'btn_segment': self.btn_segment,
            'btn_apply_refinement': self.btn_apply_refinement,
            'btn_reset_mask': self.btn_reset_mask,
            'btn_save': self.btn_save,
            'slice_slider': self.slice_slider,
            'slider_hu_min': self.slider_hu_min,
            'slider_hu_max': self.slider_hu_max,
            'slider_dilation': self.slider_dilation,
            'slider_closing': self.slider_closing,
            'cb_fill_holes': self.cb_fill_holes,
            'btn_conservative_preset': self.btn_conservative_preset,
            'btn_aggressive_preset': self.btn_aggressive_preset
        }
        self.state_manager.register_widgets(widgets)
        self.state_manager.transition_to(UIState.INITIAL)
    
    # ========== UI Helpers ==========
    
    def _toggle_ui_mode(self) -> None:
        """Переключает между упрощенным и расширенным режимами."""
        self.simplified_mode = not self.simplified_mode
        
        if self.simplified_mode:
            self.advanced_params_widget.setVisible(False)
            self.btn_toggle_mode.setText("🔧 Расширенные настройки...")
            self._add_to_history("↔️ Переключено в упрощенный режим")
        else:
            self.advanced_params_widget.setVisible(True)
            self.btn_toggle_mode.setText("⚡ Быстрые настройки...")
            self._add_to_history("↔️ Переключено в расширенный режим")
    
    def _toggle_refinement_panel(self, checked: bool) -> None:
        """Переключает видимость панели постобработки."""
        self._update_panel_positions(animate=True)
    
    def _update_panel_positions(self, animate: bool = True) -> None:
        """Обновляет позиции панелей с анимацией."""
        win_width = self.width()
        win_height = self.height()
        
        panel_config = self.config['ui']['panels']
        control_width = panel_config['control_panel_width']
        refine_width = panel_config['refine_panel_width']
        
        is_panel_open = self.toggle_refine_button.isChecked()
        
        if is_panel_open:
            refine_x = win_width - refine_width
            left_width = win_width - control_width - refine_width
            self.toggle_refine_button.setText("◀")
        else:
            refine_x = win_width
            left_width = win_width - control_width
            self.toggle_refine_button.setText("▶")
        
        control_x = left_width
        
        geo_left = QRect(0, 0, left_width, win_height)
        geo_control = QRect(control_x, 0, control_width, win_height)
        geo_refine = QRect(refine_x, 0, refine_width, win_height)
        
        self.toggle_refine_button.move(control_x - 25, win_height // 2 - 25)
        self.toggle_refine_button.raise_()
        
        if animate:
            anim_config = self.config['ui']['animation']
            duration = anim_config['duration_ms']
            
            self.left_anim = QPropertyAnimation(self.left_panel, b"geometry")
            self.left_anim.setEndValue(geo_left)
            self.left_anim.setDuration(duration)
            self.left_anim.setEasingCurve(QEasingCurve.InOutCubic)
            
            self.control_anim = QPropertyAnimation(self.right_panel, b"geometry")
            self.control_anim.setEndValue(geo_control)
            self.control_anim.setDuration(duration)
            self.control_anim.setEasingCurve(QEasingCurve.InOutCubic)
            
            self.refine_anim = QPropertyAnimation(self.refine_panel, b"geometry")
            self.refine_anim.setEndValue(geo_refine)
            self.refine_anim.setDuration(duration)
            self.refine_anim.setEasingCurve(QEasingCurve.InOutCubic)
            
            self.anim_group = QParallelAnimationGroup(self)
            self.anim_group.addAnimation(self.left_anim)
            self.anim_group.addAnimation(self.control_anim)
            self.anim_group.addAnimation(self.refine_anim)
            self.anim_group.start()
        else:
            self.left_panel.setGeometry(geo_left)
            self.right_panel.setGeometry(geo_control)
            self.refine_panel.setGeometry(geo_refine)
    
    def resizeEvent(self, event) -> None:
        """Обрабатывает изменение размера окна."""
        if hasattr(self, 'left_panel'):
            self._update_panel_positions(animate=False)
        if event:
            super().resizeEvent(event)
    
    def _update_param_labels(self) -> None:
        """Обновляет метки параметров."""
        self.label_hu_min.setText(str(self.slider_hu_min.value()))
        self.label_hu_max.setText(str(self.slider_hu_max.value()))
        self.label_dilation.setText(str(self.slider_dilation.value()))
        self.label_closing.setText(str(self.slider_closing.value()))
    
    def _save_view_state(self) -> None:
        """Сохраняет текущее состояние zoom/pan."""
        self.saved_xlim = self.ax.get_xlim()
        self.saved_ylim = self.ax.get_ylim()
    
    def _restore_view_state(self) -> None:
        """Восстанавливает zoom/pan."""
        if self.saved_xlim is not None and self.saved_ylim is not None:
            self.ax.set_xlim(self.saved_xlim)
            self.ax.set_ylim(self.saved_ylim)
    
    def _add_to_history(self, message: str) -> None:
        """Добавляет запись в историю операций."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_text.append(f"[{timestamp}] {message}")
        print(f"[{timestamp}] {message}")
    
    def _show_welcome_message(self) -> None:
        """Показывает приветственное сообщение."""
        self._add_to_history("=" * 50)
        self._add_to_history(f"🚀 {self.config['application']['window_title']}")
        self._add_to_history("=" * 50)
        self._add_to_history(f"📋 Обнаружено моделей: {len(self.model_registry.list_models())}")
        self._add_to_history("📌 Инструкция:")
        self._add_to_history("  1. Загрузите DICOM")
        self._add_to_history("  2. Нарисуйте ROI 1 и ROI 2")
        self._add_to_history("  3. Выберите модель и запустите сегментацию")
        self._add_to_history("  4. Настройте параметры постобработки")
        self._add_to_history("  5. Сохраните результат")
        self._add_to_history("=" * 50)
    
    # ========== Event Handlers - Loading ==========
    
    def _load_dicom(self) -> None:
        """Загружает DICOM серию."""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с DICOM")
        if not folder:
            return
        
        self._add_to_history(f"📂 Загрузка DICOM из {os.path.basename(folder)}")
        
        self.state_manager.transition_to(UIState.INITIAL)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_message.setText("Загрузка DICOM...")
        
        self.load_worker = DataLoadWorker(folder)
        self.load_worker.finished.connect(self._on_load_finished)
        self.load_worker.error.connect(self._on_load_error)
        self.load_worker.start()
    
    def _on_load_finished(self, volume, spacing, origin, direction, folder):
        """Обработчик успешной загрузки DICOM."""
        self.volume = volume
        self.spacing = spacing
        self.origin = origin
        self.direction = direction
        self.dicom_folder = folder
        
        self.progress_bar.setVisible(False)
        self.progress_message.setText("")
        
        self._add_to_history(
            f"✅ Volume загружен: {volume.shape}, "
            f"HU=[{volume.min():.0f}, {volume.max():.0f}]"
        )
        
        self.current_slice = volume.shape[0] // 2
        self.slice_slider.setMaximum(volume.shape[0] - 1)
        self.slice_slider.setValue(self.current_slice)
        
        self.saved_xlim = None
        self.saved_ylim = None
        
        self._update_display()
        self.state_manager.transition_to(UIState.VOLUME_LOADED)
        
        QMessageBox.information(
            self, "✅ Успех",
            f"DICOM загружен:\nРазмер: {volume.shape}\nSpacing: {spacing}"
        )
    
    def _on_load_error(self, error_msg):
        """Обработчик ошибки загрузки."""
        self.progress_bar.setVisible(False)
        self.progress_message.setText("")
        QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить:\n{error_msg}")
        self._add_to_history(f"❌ Ошибка загрузки: {error_msg}")
    
    # ========== Event Handlers - ROI ==========
    
    def _draw_roi1(self) -> None:
        """Рисует первый ROI."""
        if self.volume is None:
            return
        self._add_to_history("📐 Рисование ROI 1")
        QMessageBox.information(
            self, "ROI 1",
            "Нарисуйте прямоугольник, включающий ВСЕ лёгкие на начальном срезе"
        )
        self.roi_selector = ROISelector(self.ax, self._roi1_callback)
        self.roi_selector.connect()
    
    def _roi1_callback(self, coords: Tuple[int, int, int, int]) -> None:
        """Callback для ROI 1."""
        x_min, x_max, y_min, y_max = coords
        self.roi_manager.set_roi1(self.current_slice, x_min, x_max, y_min, y_max)
        self._add_to_history(
            f"✓ ROI 1: z={self.current_slice}, x=[{x_min}:{x_max}], y=[{y_min}:{y_max}]"
        )
        self._update_roi_info()
        self._save_view_state()
        self._update_display()
        self._restore_view_state()
        self.canvas.draw()
    
    def _draw_roi2(self) -> None:
        """Рисует второй ROI."""
        if self.volume is None or self.roi_manager.roi1_3d is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала нарисуйте ROI 1")
            return
        
        self._add_to_history("📐 Рисование ROI 2")
        QMessageBox.information(
            self, "ROI 2",
            "Нарисуйте прямоугольник на конечном срезе"
        )
        self.roi_selector = ROISelector(self.ax, self._roi2_callback)
        self.roi_selector.connect()
    
    def _roi2_callback(self, coords: Tuple[int, int, int, int]) -> None:
        """Callback для ROI 2."""
        x_min, x_max, y_min, y_max = coords
        
        _, roi1_y0, roi1_y1, roi1_x0, roi1_x1 = self.roi_manager.roi1_3d
        roi1_width = roi1_x1 - roi1_x0
        roi1_height = roi1_y1 - roi1_y0
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        x_min = max(0, center_x - roi1_width // 2)
        x_max = min(self.volume.shape[2] - 1, x_min + roi1_width)
        y_min = max(0, center_y - roi1_height // 2)
        y_max = min(self.volume.shape[1] - 1, y_min + roi1_height)
        
        self.roi_manager.set_roi2(self.current_slice, x_min, x_max, y_min, y_max)
        self._add_to_history(
            f"✓ ROI 2: z={self.current_slice}, x=[{x_min}:{x_max}], y=[{y_min}:{y_max}]"
        )
        
        self._update_roi_info()
        self.state_manager.transition_to(UIState.ROI_DEFINED)
        
        self._save_view_state()
        self._update_display()
        self._restore_view_state()
        self.canvas.draw()
    
    def _reset_roi(self) -> None:
        """Сбрасывает все ROI."""
        self.roi_manager.reset()
        self._update_roi_info()
        self.state_manager.transition_to(UIState.VOLUME_LOADED)
        self._save_view_state()
        self._update_display()
        self._restore_view_state()
        self.canvas.draw()
        self._add_to_history("↺ ROI сброшены")
    
    def _update_roi_info(self) -> None:
        """Обновляет информацию о ROI."""
        self.roi_info_label.setText(self.roi_manager.get_info_text())
    
    # ========== Event Handlers - Segmentation ==========
    
    def _segment_roi(self) -> None:
        """Запускает сегментацию."""
        if not self.roi_manager.has_both_rois():
            QMessageBox.warning(self, "Предупреждение", "Нарисуйте оба ROI")
            return
        
        try:
            roi_coords = self.roi_manager.get_combined_roi_coords(self.volume.shape)
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))
            return
        
        model_name = self.model_combo.currentText()
        use_cpu = self.cpu_checkbox.isChecked()
        
        model_instance = self.model_registry.get_model_instance(model_name, use_cpu=use_cpu)
        if model_instance is None:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель: {model_name}")
            return
        
        self._add_to_history(
            f"🧠 Запуск сегментации: {model_name}\n"
            f"   3D ROI: z=[{roi_coords[0]}:{roi_coords[1]}], "
            f"y=[{roi_coords[2]}:{roi_coords[3]}], x=[{roi_coords[4]}:{roi_coords[5]}]"
        )
        
        self.state_manager.transition_to(UIState.SEGMENTING)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.seg_worker = SegmentationWorker(
            model_instance, self.volume, self.spacing,
            self.origin, self.direction, roi_coords
        )
        self.seg_worker.finished.connect(self._on_segmentation_finished)
        self.seg_worker.error.connect(self._on_segmentation_error)
        self.seg_worker.progress.connect(self._on_segmentation_progress)
        self.seg_worker.log.connect(self._add_to_history)
        self.seg_worker.start()
    
    def _on_segmentation_progress(self, percentage: int, message: str) -> None:
        """Обновляет прогресс сегментации."""
        self.progress_bar.setValue(percentage)
        self.progress_message.setText(message)
    
    def _on_segmentation_finished(self, mask: np.ndarray, stats: Dict[str, Any]) -> None:
        """Обработчик завершения сегментации."""
        organ_key = stats.get('organ_key', 'unknown')
        
        self.base_masks[organ_key] = mask.copy()
        self.masks[organ_key] = mask.copy()
        
        if organ_key not in [self.organ_combo.itemText(i) for i in range(self.organ_combo.count())]:
            self.organ_combo.addItem(organ_key)
        self.organ_combo.setCurrentText(organ_key)
        
        self.operation_history.append({
            'timestamp': stats.get('timestamp', datetime.now()),
            'type': 'segmentation',
            'organ': organ_key,
            'stats': stats
        })
        
        self.progress_bar.setVisible(False)
        self.progress_message.setText("")
        
        mask_stats = get_mask_statistics(mask, self.spacing)
        self.label_mask_stats.setText(
            f"Орган: {organ_key}\n"
            f"Вокселей: {mask_stats['voxel_count']:,}\n"
            f"Объём: {mask_stats['volume_ml']:.1f} мл\n"
            f"Время: {stats.get('elapsed_time', 0):.1f}с"
        )
        
        self._add_to_history(
            f"✅ Сегментация завершена: {organ_key}\n"
            f"   {mask_stats['voxel_count']:,} вокселей, {mask_stats['volume_ml']:.1f} мл"
        )
        
        self.state_manager.transition_to(UIState.MASK_READY)
        self._save_view_state()
        self._update_display()
        self._restore_view_state()
        self.canvas.draw()
        
        QMessageBox.information(
            self, "✅ Сегментация готова!",
            f"Орган: {organ_key}\n"
            f"Вокселей: {mask_stats['voxel_count']:,}\n"
            f"Объём: {mask_stats['volume_ml']:.1f} мл\n"
            f"Время: {stats.get('elapsed_time', 0):.1f}с"
        )
    
    def _on_segmentation_error(self, error_msg: str) -> None:
        """Обработчик ошибки сегментации."""
        self.progress_bar.setVisible(False)
        self.progress_message.setText("")
        self.state_manager.transition_to(UIState.ROI_DEFINED)
        QMessageBox.critical(self, "Ошибка", error_msg)
        self._add_to_history(f"❌ Ошибка сегментации: {error_msg}")
    
    # ========== Event Handlers - Refinement ==========
    
    def _apply_processing_preset(self, preset_name: str) -> None:
        """Применяет пресет параметров постобработки."""
        if preset_name not in self.config['processing']['presets']:
            return
        
        preset = self.config['processing']['presets'][preset_name]
        self.slider_hu_min.setValue(preset['hu_min'])
        self.slider_hu_max.setValue(preset['hu_max'])
        self.slider_dilation.setValue(preset['dilation_iter'])
        self.slider_closing.setValue(preset['closing_size'])
        self.cb_fill_holes.setChecked(preset['fill_holes'])
        
        self._add_to_history(f"⚡ Применён пресет: {preset_name}")
    
    def _apply_mask_refinement(self) -> None:
        """Применяет постобработку к маске."""
        organ_key = self.organ_combo.currentText()
        if organ_key not in self.base_masks:
            QMessageBox.warning(self, "Предупреждение", "Сначала выполните сегментацию")
            return
        
        params = {
            'hu_min': self.slider_hu_min.value(),
            'hu_max': self.slider_hu_max.value(),
            'dilation_iter': self.slider_dilation.value(),
            'closing_size': self.slider_closing.value(),
            'fill_holes': self.cb_fill_holes.isChecked()
        }
        
        self._add_to_history(
            f"🔧 Постобработка: {organ_key}\n"
            f"   HU=[{params['hu_min']},{params['hu_max']}], "
            f"Dil={params['dilation_iter']}, Close={params['closing_size']}"
        )
        
        self.state_manager.transition_to(UIState.REFINING)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        
        self.refine_worker = MaskRefinementWorker(
            self.base_masks[organ_key], self.volume, self.spacing, params, organ_key
        )
        self.refine_worker.finished.connect(self._on_refinement_finished)
        self.refine_worker.error.connect(self._on_refinement_error)
        self.refine_worker.progress.connect(self._on_segmentation_progress)
        self.refine_worker.start()
    
    def _on_refinement_finished(self, mask: np.ndarray, stats: Dict[str, Any]) -> None:
        """Обработчик завершения постобработки."""
        organ_key = stats.get('organ_key', self.organ_combo.currentText())
        self.masks[organ_key] = mask
        
        self.operation_history.append({
            'timestamp': stats.get('timestamp', datetime.now()),
            'type': 'refinement',
            'organ': organ_key,
            'stats': stats
        })
        
        self.progress_bar.setVisible(False)
        self.progress_message.setText("")
        
        improvement = stats.get('improvement_percent', 0)
        improvement_sign = "+" if improvement > 0 else ""
        self.label_mask_stats.setText(
            f"Орган: {organ_key}\n"
            f"Базовая: {stats['base_count']:,} вокселей\n"
            f"Текущая: {stats['final_count']:,} ({improvement_sign}{improvement:.1f}%)\n"
            f"Объём: {stats['volume_ml']:.1f} мл"
        )
        
        self._add_to_history(
            f"✅ Постобработка завершена: {organ_key}\n"
            f"   {stats['final_count']:,} вокселей ({improvement_sign}{improvement:.1f}%)"
        )
        
        self.state_manager.transition_to(UIState.MASK_READY)
        self._save_view_state()
        self._update_display()
        self._restore_view_state()
        self.canvas.draw()
    
    def _on_refinement_error(self, error_msg: str) -> None:
        """Обработчик ошибки постобработки."""
        self.progress_bar.setVisible(False)
        self.progress_message.setText("")
        self.state_manager.transition_to(UIState.MASK_READY)
        QMessageBox.critical(self, "Ошибка", error_msg)
        self._add_to_history(f"❌ Ошибка постобработки: {error_msg}")
    
    def _reset_to_base_mask(self) -> None:
        """Сбрасывает маску к базовой."""
        organ_key = self.organ_combo.currentText()
        if organ_key in self.base_masks:
            self.masks[organ_key] = self.base_masks[organ_key].copy()
            self._add_to_history(f"↺ Сброс к базовой маске: {organ_key}")
            self._save_view_state()
            self._update_display()
            self._restore_view_state()
            self.canvas.draw()
    
    # ========== Event Handlers - Display ==========
    
    def _toggle_auto_window(self, state: bool) -> None:
        """Переключает автоматический window/level."""
        self.auto_window = state
        self.spin_wc.setEnabled(not state)
        self.spin_ww.setEnabled(not state)
        self.btn_lung_preset.setEnabled(not state)
        self.btn_mediastinum_preset.setEnabled(not state)
        if self.volume is not None:
            self._save_view_state()
            self._update_display()
            self._restore_view_state()
            self.canvas.draw()
    
    def _on_window_changed(self) -> None:
        """Обработчик изменения window/level."""
        self.window_center = self.spin_wc.value()
        self.window_width = self.spin_ww.value()
        if self.volume is not None:
            self._save_view_state()
            self._update_display()
            self._restore_view_state()
            self.canvas.draw()
    
    def _apply_window_preset(self, preset_name: str) -> None:
        """Применяет пресет window/level."""
        if preset_name not in self.config['display']['presets']:
            return
        preset = self.config['display']['presets'][preset_name]
        self.cb_auto_window.setChecked(False)
        self.spin_wc.setValue(preset['center'])
        self.spin_ww.setValue(preset['width'])
    
    def _on_slice_changed(self, value: int) -> None:
        """Обработчик изменения среза."""
        self.current_slice = value
        self._save_view_state()
        self._update_display()
        self._restore_view_state()
        self.canvas.draw()
    
    def _change_view(self, view: str) -> None:
        """Меняет проекцию просмотра."""
        self.current_view = view
        max_slice = self._get_max_slice()
        self.slice_slider.setMaximum(max_slice)
        self.current_slice = min(self.current_slice, max_slice)
        self.slice_slider.setValue(self.current_slice)
        self.saved_xlim = None
        self.saved_ylim = None
        self._update_display()
        self.canvas.draw()
    
    def _update_display(self) -> None:
        """Обновляет отображение изображения."""
        if self.volume is None:
            return
        
        self.ax.clear()
        slice_data = self._get_slice_data()
        vmin, vmax = self._apply_window_level(slice_data)
        
        self.ax.imshow(slice_data, cmap='gray', aspect='equal',
                      interpolation='nearest', vmin=vmin, vmax=vmax)
        
        # Отрисовка масок
        for organ_key, mask in self.masks.items():
            mask_slice = self._get_mask_slice(mask)
            if mask_slice is not None and np.any(mask_slice):
                color = self.config['organ_colors'].get(organ_key, 
                                                         self.config['organ_colors']['default'])
                self.ax.contour(mask_slice, levels=[0.5], colors=color, linewidths=2)
        
        # Отрисовка ROI
        self._draw_roi_boxes()
        
        max_slice = self._get_max_slice()
        self.slice_label.setText(f"Slice: {self.current_slice}/{max_slice}")
        
        wl_text = f"WL={self.window_center}, WW={self.window_width}" if not self.auto_window else "Auto"
        self.ax.set_title(f"{self.current_view.capitalize()} - Slice {self.current_slice} ({wl_text})")
    
    def _get_slice_data(self) -> np.ndarray:
        """Возвращает данные текущего среза."""
        if self.current_view == 'axial':
            return self.volume[self.current_slice, :, :]
        elif self.current_view == 'coronal':
            return self.volume[:, self.current_slice, :]
        else:
            return self.volume[:, :, self.current_slice]
    
    def _get_mask_slice(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Возвращает срез маски."""
        if self.current_view == 'axial':
            return mask[self.current_slice, :, :]
        elif self.current_view == 'coronal':
            return mask[:, self.current_slice, :]
        else:
            return mask[:, :, self.current_slice]
    
    def _get_max_slice(self) -> int:
        """Возвращает максимальный номер среза."""
        if self.current_view == 'axial':
            return self.volume.shape[0] - 1
        elif self.current_view == 'coronal':
            return self.volume.shape[1] - 1
        else:
            return self.volume.shape[2] - 1
    
    def _apply_window_level(self, image: np.ndarray) -> Tuple[float, float]:
        """Применяет window/level к изображению."""
        if self.auto_window:
            vmin = float(np.percentile(image, 2))
            vmax = float(np.percentile(image, 98))
        else:
            vmin = self.window_center - self.window_width / 2
            vmax = self.window_center + self.window_width / 2
        return vmin, vmax
    
    def _draw_roi_boxes(self) -> None:
        """Отрисовывает ROI на текущем срезе."""
        if self.roi_manager.roi1_3d is not None:
            self._draw_single_roi(self.roi_manager.roi1_3d, 'red', 'ROI1')
        if self.roi_manager.roi2_3d is not None:
            self._draw_single_roi(self.roi_manager.roi2_3d, 'blue', 'ROI2')
    
    def _draw_single_roi(self, roi_3d: Tuple[int, int, int, int, int], 
                         color: str, label: str) -> None:
        """Отрисовывает один ROI."""
        z_slice, y0, y1, x0, x1 = roi_3d
        
        if self.current_view == 'axial':
            if z_slice == self.current_slice:
                rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2,
                                edgecolor=color, facecolor='none', linestyle='-')
                self.ax.add_patch(rect)
                self.ax.text(x0, y0-5, label, color=color, fontsize=10, weight='bold')
            else:
                rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2,
                                 edgecolor=color, facecolor='none', linestyle='--')
                self.ax.add_patch(rect)