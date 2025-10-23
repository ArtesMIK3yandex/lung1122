"""
Виджет статусной строки с отображением состояния системы
"""

from PyQt5.QtWidgets import QStatusBar, QLabel, QProgressBar
from PyQt5.QtCore import Qt


class StatusWidget(QStatusBar):
    """Виджет статусной строки с отображением состояния"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Основная метка состояния
        self.status_label = QLabel("Готов к работе")
        self.addWidget(self.status_label)
        
        # Прогресс-бар (скрыт по умолчанию)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.addPermanentWidget(self.progress_bar)
        
        # Контекстные подсказки
        self.hint_label = QLabel("")
        self.hint_label.setStyleSheet("color: #888; font-style: italic;")
        self.addPermanentWidget(self.hint_label)
    
    def set_status(self, message: str):
        """Устанавливает текст состояния"""
        self.status_label.setText(message)
    
    def set_hint(self, hint: str):
        """Устанавливает контекстную подсказку"""
        self.hint_label.setText(hint)
    
    def show_progress(self, value: int = 0, maximum: int = 100):
        """Показывает прогресс-бар"""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(True)
    
    def hide_progress(self):
        """Скрывает прогресс-бар"""
        self.progress_bar.setVisible(False)