"""
Панель администратора (показывается только в режиме Admin)
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox, QMessageBox


class AdminPanel(QWidget):
    """
    Панель администратора (показывается только в режиме Admin).
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_manager = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Настройка UI"""
        layout = QVBoxLayout(self)
        
        # Группа администратора
        admin_group = QGroupBox("⚙️ Администрирование")
        admin_layout = QVBoxLayout()
        
        # Кнопка настроек
        config_btn = QPushButton("Настройки конфигурации")
        config_btn.clicked.connect(self._on_config_clicked)
        admin_layout.addWidget(config_btn)
        
        # Кнопка управления модулями
        modules_btn = QPushButton("Управление модулями")
        modules_btn.clicked.connect(self._on_modules_clicked)
        admin_layout.addWidget(modules_btn)
        
        # Кнопка компоновки
        layout_btn = QPushButton("Настройка компоновки")
        layout_btn.clicked.connect(self._on_layout_clicked)
        admin_layout.addWidget(layout_btn)
        
        admin_group.setLayout(admin_layout)
        layout.addWidget(admin_group)
    
    def set_config_manager(self, config_manager):
        """Устанавливает менеджер конфигурации"""
        self.config_manager = config_manager
    
    def _on_config_clicked(self):
        """Открытие редактора конфигурации"""
        from gui.dialogs.config_editor import ConfigEditorDialog
        
        if self.config_manager:
            dialog = ConfigEditorDialog(self.config_manager, self)
            dialog.exec_()
        else:
            QMessageBox.warning(self, "Ошибка", "ConfigManager не установлен")
    
    def _on_modules_clicked(self):
        """Управление модулями"""
        QMessageBox.information(self, "Модули", "Функция в разработке")
    
    def _on_layout_clicked(self):
        """Настройка компоновки"""
        QMessageBox.information(self, "Компоновка", "Функция в разработке")