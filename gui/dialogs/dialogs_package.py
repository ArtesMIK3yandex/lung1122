"""
Диалоговые окна приложения:
- LoginDialog: вход администратора
- SeriesSelectorDialog: выбор серии DICOM
- ConfigEditorDialog: редактирование конфигурации (для админа)
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QListWidget, QMessageBox,
                             QGroupBox, QComboBox, QSpinBox, QCheckBox, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget

# ===== login_dialog.py =====

class LoginDialog(QDialog):
    """Диалог входа администратора"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Вход администратора")
        self.setModal(True)
        self.password = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Инструкция
        info_label = QLabel("Введите пароль администратора:")
        layout.addWidget(info_label)
        
        # Поле пароля
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self.accept)
        layout.addWidget(self.password_input)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        ok_btn = QPushButton("Войти")
        ok_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setFixedWidth(300)
    
    def get_password(self) -> str:
        """Возвращает введенный пароль"""
        return self.password_input.text()


# ===== series_selector.py =====

class SeriesSelectorDialog(QDialog):
    """Диалог выбора серии DICOM"""
    
    def __init__(self, series_list: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор серии DICOM")
        self.setModal(True)
        self.series_list = series_list  # [(uid, description), ...]
        self.selected_uid = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Инструкция
        info_label = QLabel("Найдено несколько серий. Выберите серию для загрузки:")
        layout.addWidget(info_label)
        
        # Список серий
        self.series_listwidget = QListWidget()
        for uid, description in self.series_list:
            self.series_listwidget.addItem(description)
        
        if self.series_list:
            self.series_listwidget.setCurrentRow(0)
        
        self.series_listwidget.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.series_listwidget)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        ok_btn = QPushButton("Загрузить")
        ok_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addLayout(buttons_layout)
        
        self.resize(500, 300)
    
    def get_selected_series(self) -> str:
        """Возвращает UID выбранной серии"""
        current_row = self.series_listwidget.currentRow()
        if 0 <= current_row < len(self.series_list):
            return self.series_list[current_row][0]
        return None


# ===== config_editor.py =====

class ConfigEditorDialog(QDialog):
    """Редактор конфигурации для администратора"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки администратора")
        self.setModal(True)
        self.config_manager = config_manager
        
        self._setup_ui()
        self.resize(700, 500)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Вкладки
        tabs = QTabWidget()
        
        # Вкладка 1: Режимы обработки
        processing_tab = self._create_processing_modes_tab()
        tabs.addTab(processing_tab, "Режимы обработки")
        
        # Вкладка 2: Видимость модулей
        modules_tab = self._create_modules_tab()
        tabs.addTab(modules_tab, "Модули")
        
        # Вкладка 3: Компоновка интерфейса
        layout_tab = self._create_layout_tab()
        tabs.addTab(layout_tab, "Компоновка")
        
        layout.addWidget(tabs)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self._on_save)
        buttons_layout.addWidget(save_btn)
        
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(close_btn)
        
        layout.addLayout(buttons_layout)
    
    def _create_processing_modes_tab(self):
        """Вкладка управления режимами обработки"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Таблица режимов
        self.modes_table = QTableWidget()
        self.modes_table.setColumnCount(3)
        self.modes_table.setHorizontalHeaderLabels(["ID", "Название", "Параметры"])
        self.modes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Загрузка режимов
        self._load_processing_modes()
        
        layout.addWidget(self.modes_table)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        add_btn = QPushButton("Добавить")
        add_btn.clicked.connect(self._on_add_mode)
        buttons_layout.addWidget(add_btn)
        
        edit_btn = QPushButton("Редактировать")
        edit_btn.clicked.connect(self._on_edit_mode)
        buttons_layout.addWidget(edit_btn)
        
        delete_btn = QPushButton("Удалить")
        delete_btn.clicked.connect(self._on_delete_mode)
        buttons_layout.addWidget(delete_btn)
        
        layout.addLayout(buttons_layout)
        
        return widget
    
    def _load_processing_modes(self):
        """Загружает режимы обработки в таблицу"""
        modes = self.config_manager.get_processing_modes()
        self.modes_table.setRowCount(len(modes))
        
        for i, mode in enumerate(modes):
            self.modes_table.setItem(i, 0, QTableWidgetItem(mode['id']))
            self.modes_table.setItem(i, 1, QTableWidgetItem(mode['name']))
            self.modes_table.setItem(i, 2, QTableWidgetItem(str(mode['parameters'])))
    
    def _create_modules_tab(self):
        """Вкладка управления модулями"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Таблица модулей
        self.modules_table = QTableWidget()
        self.modules_table.setColumnCount(4)
        self.modules_table.setHorizontalHeaderLabels(["ID", "Название", "Видимость", "Порядок"])
        self.modules_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Загрузка модулей
        self._load_modules()
        
        layout.addWidget(self.modules_table)
        
        return widget
    
    def _load_modules(self):
        """Загружает модули в таблицу"""
        modules = self.config_manager.get_modules()
        self.modules_table.setRowCount(len(modules))
        
        for i, (module_id, config) in enumerate(modules.items()):
            self.modules_table.setItem(i, 0, QTableWidgetItem(module_id))
            self.modules_table.setItem(i, 1, QTableWidgetItem(config['name']))
            
            visible_item = QTableWidgetItem("Да" if config['visible'] else "Нет")
            self.modules_table.setItem(i, 2, visible_item)
            
            self.modules_table.setItem(i, 3, QTableWidgetItem(str(config['order'])))
    
    def _create_layout_tab(self):
        """Вкладка управления компоновкой"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel("Настройка порядка элементов интерфейса")
        layout.addWidget(info_label)
        
        # TODO: Реализация drag-and-drop для перестановки элементов
        placeholder = QLabel("Функционал в разработке")
        placeholder.setAlignment(Qt.AlignCenter)
        layout.addWidget(placeholder)
        
        return widget
    
    def _on_add_mode(self):
        """Добавление нового режима"""
        # Простой диалог для демонстрации
        from PyQt5.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(self, "Новый режим", "Введите название:")
        if ok and name:
            mode_id = self.config_manager.add_processing_mode(name, {})
            QMessageBox.information(self, "Успех", f"Режим '{name}' добавлен")
            self._load_processing_modes()
    
    def _on_edit_mode(self):
        """Редактирование режима"""
        current_row = self.modes_table.currentRow()
        if current_row >= 0:
            QMessageBox.information(self, "Информация", "Функция редактирования в разработке")
    
    def _on_delete_mode(self):
        """Удаление режима"""
        current_row = self.modes_table.currentRow()
        if current_row >= 0:
            mode_id = self.modes_table.item(current_row, 0).text()
            
            reply = QMessageBox.question(
                self, "Подтверждение", 
                f"Удалить режим '{mode_id}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.config_manager.delete_processing_mode(mode_id)
                self._load_processing_modes()
    
    def _on_save(self):
        """Сохранение изменений"""
        self.config_manager.save()
        QMessageBox.information(self, "Успех", "Конфигурация сохранена")
