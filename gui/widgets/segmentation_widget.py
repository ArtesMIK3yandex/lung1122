"""
Виджет сегментации - НЕОТКЛЮЧАЕМЫЙ модуль
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox
from PyQt5.QtCore import Qt, pyqtSignal


class SegmentationWidget(QWidget):
    """
    Виджет сегментации - НЕОТКЛЮЧАЕМЫЙ модуль (отдельный блок).
    Всегда видим пользователю.
    """
    
    segmentation_requested = pyqtSignal(dict)  # Сигнал запуска сегментации
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Настройка UI"""
        layout = QVBoxLayout(self)
        
        # Группа сегментации
        seg_group = QGroupBox("Сегментация лёгких")
        seg_layout = QVBoxLayout()
        
        # Информация
        info_label = QLabel("Автоматическая сегментация лёгких на выбранном диапазоне срезов")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888;")
        seg_layout.addWidget(info_label)
        
        # Кнопка запуска
        self.run_btn = QPushButton("Запустить сегментацию")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.run_btn.setMinimumHeight(40)
        seg_layout.addWidget(self.run_btn)
        
        # Статус
        self.status_label = QLabel("Готов к запуску")
        self.status_label.setAlignment(Qt.AlignCenter)
        seg_layout.addWidget(self.status_label)
        
        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)
    
    def set_logger(self, logger):
        """Устанавливает логгер"""
        self.logger = logger
    
    def _on_run_clicked(self):
        """Обработка запуска сегментации"""
        self.status_label.setText("Сегментация запущена...")
        self.run_btn.setEnabled(False)
        
        # Отправляем сигнал для запуска сегментации
        self.segmentation_requested.emit({})
        
        if self.logger:
            self.logger.log_segmentation_run()
    
    def on_segmentation_complete(self, success: bool = True):
        """Callback после завершения сегментации"""
        if success:
            self.status_label.setText("✓ Сегментация завершена")
        else:
            self.status_label.setText("✗ Ошибка сегментации")
        
        self.run_btn.setEnabled(True)