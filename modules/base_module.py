"""
Базовый класс для всех подключаемых модулей.
Обеспечивает единый интерфейс для динамической загрузки.
"""

from PyQt5.QtWidgets import QWidget
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModule(ABC, QWidget):
    """
    Абстрактный базовый класс для всех модулей приложения
    
    Каждый модуль должен наследоваться от этого класса и реализовать:
    - get_module_info(): метаданные модуля
    - initialize(): инициализация UI и логики
    - process(): основная логика обработки
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = None
        self.config_manager = None
        self._initialized = False
    
    @abstractmethod
    def get_module_info(self) -> Dict[str, Any]:
        """
        Возвращает метаданные модуля
        
        Returns:
            {
                'id': 'unique_module_id',
                'name': 'Отображаемое имя',
                'version': '1.0.0',
                'description': 'Описание функционала',
                'removable': True/False  # Можно ли скрыть модуль
            }
        """
        pass
    
    @abstractmethod
    def initialize(self):
        """
        Инициализация модуля: создание UI, подключение сигналов
        Вызывается один раз при загрузке модуля
        """
        pass
    
    def set_logger(self, logger):
        """Устанавливает логгер для модуля"""
        self.logger = logger
    
    def set_config_manager(self, config_manager):
        """Устанавливает менеджер конфигурации"""
        self.config_manager = config_manager
    
    def is_initialized(self) -> bool:
        """Проверяет, инициализирован ли модуль"""
        return self._initialized
    
    def on_data_loaded(self, data):
        """
        Callback при загрузке новых данных
        Переопределяется в дочерних классах при необходимости
        """
        pass
    
    def on_processing_complete(self, result):
        """
        Callback при завершении обработки
        Переопределяется в дочерних классах при необходимости
        """
        pass
    
    def cleanup(self):
        """
        Очистка ресурсов модуля
        Переопределяется в дочерних классах при необходимости
        """
        pass