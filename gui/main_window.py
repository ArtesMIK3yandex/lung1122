"""
Главное окно приложения - выступает только как компоновщик виджетов.
Минимальная логика, максимальная модульность.
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QMenuBar, QMenu, QAction, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent

from core.auth import AuthManager
from core.config_manager import ConfigManager
from core.logger import ActionLogger
from core.dicom_loader import DICOMLoader
from utils.plugin_loader import PluginLoader

from gui.widgets.projection_manager import ProjectionManager
from gui.widgets.viewer_widget import ViewerWidget
from gui.widgets.status_widget import StatusWidget
from gui.dialogs.login_dialog import LoginDialog
from gui.dialogs.series_selector import SeriesSelectorDialog


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    # Сигналы
    data_loaded = pyqtSignal(object)  # Сигнал загрузки данных
    role_changed = pyqtSignal(str)     # Сигнал изменения роли
    
    def __init__(self):
        super().__init__()
        
        # Инициализация core-компонентов
        self.auth_manager = AuthManager()
        self.config_manager = ConfigManager()
        self.logger = ActionLogger()
        self.dicom_loader = DICOMLoader()
        self.plugin_loader = PluginLoader()
        
        # Загрузка плагинов
        self.plugin_loader.discover_modules()
        
        # Регистрация модулей в конфигурации
        self._register_discovered_modules()
        
        # UI компоненты
        self.projection_manager = None
        self.viewer_widget = None
        self.status_widget = None
        self.modules_widgets = {}  # {module_id: widget_instance}
        
        self._setup_ui()
        self._setup_drag_drop()
        self._connect_signals()
        
        self.setWindowTitle("lung1122 - Medical Imaging Viewer")
        self.resize(1400, 900)
    
    def _setup_ui(self):
        """Настройка интерфейса"""
        # Меню
        self._create_menu_bar()
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Основной сплиттер (левая панель | центр | правая панель)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # === ЛЕВАЯ ПАНЕЛЬ (управление) ===
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # === ЦЕНТР (проекции) ===
        self.projection_manager = ProjectionManager()
        self.projection_manager.set_dicom_loader(self.dicom_loader)
        main_splitter.addWidget(self.projection_manager)
        
        # === ПРАВАЯ ПАНЕЛЬ (модули) ===
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Пропорции: 1:3:1
        main_splitter.setSizes([250, 800, 250])
        
        main_layout.addWidget(main_splitter)
        
        # Статусбар внизу
        self.status_widget = StatusWidget()
        self.setStatusBar(self.status_widget)
    
    def _create_menu_bar(self):
        """Создание меню"""
        menubar = self.menuBar()
        
        # Файл
        file_menu = menubar.addMenu("Файл")
        
        load_action = QAction("Загрузить DICOM...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._on_load_dicom_clicked)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Режим
        mode_menu = menubar.addMenu("Режим")
        
        admin_action = QAction("Войти как Администратор", self)
        admin_action.triggered.connect(self._on_admin_login)
        mode_menu.addAction(admin_action)
        
        user_action = QAction("Выйти из режима Администратора", self)
        user_action.triggered.connect(self._on_admin_logout)
        mode_menu.addAction(user_action)
    
    def _create_left_panel(self) -> QWidget:
        """Создание левой панели управления"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # ViewerWidget (базовый функционал просмотра - НЕОТКЛЮЧАЕМЫЙ)
        self.viewer_widget = ViewerWidget()
        self.viewer_widget.set_dicom_loader(self.dicom_loader)
        self.viewer_widget.set_logger(self.logger)
        layout.addWidget(self.viewer_widget)
        
        layout.addStretch()
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Создание правой панели (модули)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Загрузка видимых модулей из конфигурации
        self._load_visible_modules(layout)
        
        layout.addStretch()
        return panel
    
    def _load_visible_modules(self, layout):
        """Загружает видимые модули в layout"""
        visible_modules = self.config_manager.get_visible_modules()
        
        # Сортируем по порядку
        sorted_modules = sorted(
            visible_modules.items(), 
            key=lambda x: x[1].get('order', 999)
        )
        
        for module_id, config in sorted_modules:
            module_widget = self.plugin_loader.instantiate_module(module_id, self)
            
            if module_widget:
                module_widget.set_logger(self.logger)
                module_widget.set_config_manager(self.config_manager)
                module_widget.initialize()
                
                self.modules_widgets[module_id] = module_widget
                layout.addWidget(module_widget)
    
    def _register_discovered_modules(self):
        """Регистрирует обнаруженные модули в конфигурации"""
        available_modules = self.plugin_loader.get_available_modules()
        
        for module_info in available_modules:
            module_id = module_info['id']
            
            # Проверяем, зарегистрирован ли уже
            if module_id not in self.config_manager.get_modules():
                self.config_manager.register_module(module_id, {
                    'name': module_info['name'],
                    'visible': True,
                    'order': 999,
                    'removable': module_info.get('removable', True)
                })
    
    def _setup_drag_drop(self):
        """Настройка Drag & Drop для загрузки DICOM"""
        self.setAcceptDrops(True)
    
    def _connect_signals(self):
        """Подключение сигналов"""
        self.data_loaded.connect(self._on_data_loaded)
        self.role_changed.connect(self._on_role_changed)
    
    # === DRAG & DROP ===
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Обработка перетаскивания"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Обработка сброса файлов/папок"""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._load_dicom_from_path(path)
    
    # === ОБРАБОТЧИКИ СОБЫТИЙ ===
    
    def _on_load_dicom_clicked(self):
        """Обработчик кнопки загрузки DICOM"""
        from PyQt5.QtWidgets import QFileDialog
        from pathlib import Path
        
        path = QFileDialog.getExistingDirectory(self, "Выберите папку с DICOM")
        if path:
            self._load_dicom_from_path(path)
    
    def _load_dicom_from_path(self, path: str):
        """Загрузка DICOM из указанного пути"""
        from pathlib import Path
        
        self.status_widget.set_status("Сканирование DICOM-файлов...")
        
        path_obj = Path(path)
        series_dict = self.dicom_loader.scan_directory(path_obj, recursive=True)
        
        if not series_dict:
            QMessageBox.warning(self, "Ошибка", "DICOM-файлы не найдены")
            self.status_widget.set_status("Ошибка: файлы не найдены")
            return
        
        # Если одна серия - загружаем автоматически
        if len(series_dict) == 1:
            series_uid = list(series_dict.keys())[0]
            self._load_series(series_uid)
        else:
            # Множество серий - показываем диалог выбора
            dialog = SeriesSelectorDialog(self.dicom_loader.get_series_list(), self)
            if dialog.exec_():
                selected_uid = dialog.get_selected_series()
                if selected_uid:
                    self._load_series(selected_uid)
    
    def _load_series(self, series_uid: str):
        """Загрузка выбранной серии"""
        self.status_widget.set_status("Загрузка серии...")
        
        if self.dicom_loader.load_series(series_uid):
            self.logger.log_dicom_load(series_uid, series_uid)
            self.status_widget.set_status("Серия загружена. Готов к работе.")
            
            # Уведомляем все модули о загрузке данных
            self.data_loaded.emit(self.dicom_loader)
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить серию")
            self.status_widget.set_status("Ошибка загрузки")
    
    def _on_data_loaded(self, loader):
        """Обработка загрузки новых данных"""
        # Обновляем проекции
        self.projection_manager.update_views()
        
        # Уведомляем модули
        for module_widget in self.modules_widgets.values():
            module_widget.on_data_loaded(loader)
    
    def _on_admin_login(self):
        """Вход в режим администратора"""
        dialog = LoginDialog(self)
        if dialog.exec_():
            password = dialog.get_password()
            if self.auth_manager.login_admin(password):
                self.logger.set_role("Admin")
                self.role_changed.emit("Admin")
                QMessageBox.information(self, "Успех", "Вход выполнен как Администратор")
            else:
                QMessageBox.warning(self, "Ошибка", "Неверный пароль")
    
    def _on_admin_logout(self):
        """Выход из режима администратора"""
        self.auth_manager.logout()
        self.logger.set_role("User")
        self.role_changed.emit("User")
        QMessageBox.information(self, "Информация", "Выход из режима Администратора")
    
    def _on_role_changed(self, role: str):
        """Обработка изменения роли"""
        # Здесь можно добавить логику показа/скрытия элементов админа
        pass
