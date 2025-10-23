"""
Виджет обработки маски с предустановленными режимами
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QGroupBox, QMessageBox)
from PyQt5.QtCore import pyqtSignal


class ProcessingWidget(QWidget):
    """
    Виджет обработки маски с предустановленными режимами.
    """
    
    processing_requested = pyqtSignal(str, dict)  # mode_id, parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_manager = None
        self.logger = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Настройка UI"""
        layout = QVBoxLayout(self)
        
        # Группа обработки
        proc_group = QGroupBox("Обработка маски")
        proc_layout = QVBoxLayout()
        
        # Выпадающий список режимов
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Режим:"))
        self.mode_combo = QComboBox()
        mode_layout.addWidget(self.mode_combo)
        proc_layout.addLayout(mode_layout)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.confirm_btn = QPushButton("Подтвердить выполнение")
        self.confirm_btn.clicked.connect(self._on_confirm_clicked)
        buttons_layout.addWidget(self.confirm_btn)
        
        self.reset_btn = QPushButton("Сброс")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        buttons_layout.addWidget(self.reset_btn)
        
        proc_layout.addLayout(buttons_layout)
        
        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)
    
    def set_config_manager(self, config_manager):
        """Устанавливает менеджер конфигурации"""
        self.config_manager = config_manager
        self._load_modes()
    
    def set_logger(self, logger):
        """Устанавливает логгер"""
        self.logger = logger
    
    def _load_modes(self):
        """Загружает режимы из конфигурации"""
        if self.config_manager is None:
            return
        
        self.mode_combo.clear()
        modes = self.config_manager.get_processing_modes()
        
        for mode in modes:
            self.mode_combo.addItem(mode['name'], mode)
    
    def _on_confirm_clicked(self):
        """Обработка подтверждения выполнения"""
        current_mode = self.mode_combo.currentData()
        
        if current_mode:
            mode_id = current_mode['id']
            parameters = current_mode['parameters']
            
            self.processing_requested.emit(mode_id, parameters)
            
            if self.logger:
                self.logger.log_processing_mode(current_mode['name'], parameters)
            
            QMessageBox.information(self, "Обработка", 
                                  f"Запущена обработка в режиме: {current_mode['name']}")
    
    def _on_reset_clicked(self):
        """Обработка сброса"""
        reply = QMessageBox.question(
            self, "Сброс", 
            "Сбросить обработку?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.logger:
                self.logger.log_action("reset_processing")
            QMessageBox.information(self, "Сброс", "Обработка сброшена")