"""
Динамический загрузчик моделей сегментации.

Автоматически обнаруживает и загружает модели из папки models/,
позволяя добавлять новые модели без изменения основного кода.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, List, Optional

# Импортируем базовый класс
# В реальном проекте это будет: from models.base_model import BaseSegmenter
# Здесь используем типизацию для примера
from typing import Protocol


class BaseSegmenterProtocol(Protocol):
    """Протокол для базового класса сегментации."""
    display_name: str
    organ_key: str
    
    def segment(self, *args, **kwargs):
        """Метод сегментации."""
        ...


def discover_models(models_dir: str = "models") -> Dict[str, Type]:
    """
    Автоматически обнаруживает все доступные модели сегментации.
    
    Сканирует папку models/ и находит все классы, наследующие BaseSegmenter.
    
    Args:
        models_dir: Путь к папке с моделями (относительно корня проекта)
        
    Returns:
        Словарь {display_name: Class} со всеми найденными моделями
        
    Example:
        >>> models = discover_models()
        >>> print(models.keys())
        dict_keys(['Сегментация лёгких', 'Сегментация печени'])
        
        >>> lung_model = models['Сегментация лёгких']()
        >>> mask = lung_model.segment(volume, spacing)
    """
    models = {}
    
    # Получаем абсолютный путь к папке models
    if not os.path.isabs(models_dir):
        # Предполагаем, что запуск из корня проекта
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, models_dir)
    
    if not os.path.exists(models_dir):
        print(f"[WARNING] Папка моделей не найдена: {models_dir}")
        return models
    
    # Добавляем папку в PYTHONPATH для импорта
    if models_dir not in sys.path:
        sys.path.insert(0, os.path.dirname(models_dir))
    
    # Импортируем базовый класс
    try:
        base_module = importlib.import_module("models.base_model")
        BaseSegmenter = getattr(base_module, "BaseSegmenter")
    except (ImportError, AttributeError) as e:
        print(f"[ERROR] Не удалось импортировать BaseSegmenter: {e}")
        return models
    
    # Сканируем все .py файлы в папке models
    for filename in os.listdir(models_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]  # Убираем .py
            
            try:
                # Импортируем модуль
                module = importlib.import_module(f"models.{module_name}")
                
                # Ищем классы, наследующие BaseSegmenter
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseSegmenter) and 
                        obj is not BaseSegmenter and
                        hasattr(obj, 'display_name') and 
                        hasattr(obj, 'organ_key')):
                        
                        # Используем display_name как ключ
                        display_name = obj.display_name
                        models[display_name] = obj
                        print(f"[INFO] Обнаружена модель: {display_name} ({module_name})")
            
            except Exception as e:
                print(f"[WARNING] Не удалось загрузить модуль {module_name}: {e}")
                continue
    
    print(f"[INFO] Всего обнаружено моделей: {len(models)}")
    return models


def get_model_info(model_class: Type) -> Dict[str, str]:
    """
    Извлекает информацию о модели.
    
    Args:
        model_class: Класс модели
        
    Returns:
        Словарь с информацией о модели
        
    Example:
        >>> info = get_model_info(LungMaskSegmenter)
        >>> print(info['display_name'])
        'Сегментация лёгких'
    """
    return {
        'display_name': getattr(model_class, 'display_name', 'Unknown'),
        'organ_key': getattr(model_class, 'organ_key', 'unknown'),
        'description': getattr(model_class, '__doc__', 'No description'),
        'module': model_class.__module__,
        'class_name': model_class.__name__
    }


def list_available_models(models_dict: Dict[str, Type]) -> List[str]:
    """
    Возвращает список названий доступных моделей для UI.
    
    Args:
        models_dict: Словарь моделей из discover_models()
        
    Returns:
        Отсортированный список display_name
        
    Example:
        >>> models = discover_models()
        >>> names = list_available_models(models)
        >>> print(names)
        ['Сегментация лёгких', 'Сегментация печени']
    """
    return sorted(models_dict.keys())


def instantiate_model(model_class: Type, **kwargs) -> object:
    """
    Создает экземпляр модели с заданными параметрами.
    
    Args:
        model_class: Класс модели
        **kwargs: Параметры для конструктора
        
    Returns:
        Экземпляр модели
        
    Example:
        >>> models = discover_models()
        >>> lung_segmenter = instantiate_model(
        ...     models['Сегментация лёгких'],
        ...     use_cpu=False
        ... )
    """
    try:
        return model_class(**kwargs)
    except Exception as e:
        print(f"[ERROR] Не удалось создать экземпляр {model_class.__name__}: {e}")
        raise


def validate_model_class(model_class: Type) -> bool:
    """
    Проверяет, что класс модели имеет все необходимые атрибуты и методы.
    
    Args:
        model_class: Класс модели для проверки
        
    Returns:
        True если валидный, False иначе
    """
    required_attributes = ['display_name', 'organ_key']
    required_methods = ['segment']
    
    # Проверяем атрибуты
    for attr in required_attributes:
        if not hasattr(model_class, attr):
            print(f"[ERROR] Модель {model_class.__name__} не имеет атрибута {attr}")
            return False
    
    # Проверяем методы
    for method in required_methods:
        if not hasattr(model_class, method) or not callable(getattr(model_class, method)):
            print(f"[ERROR] Модель {model_class.__name__} не имеет метода {method}")
            return False
    
    return True


class ModelRegistry:
    """
    Реестр моделей для удобного управления.
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.load_models()
        >>> lung_model = registry.get_model('Сегментация лёгких')
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Инициализирует реестр моделей.
        
        Args:
            models_dir: Путь к папке с моделями
        """
        self.models_dir = models_dir
        self._models: Dict[str, Type] = {}
        self._instances: Dict[str, object] = {}
    
    def load_models(self) -> None:
        """Загружает все доступные модели."""
        self._models = discover_models(self.models_dir)
    
    def get_model_class(self, display_name: str) -> Optional[Type]:
        """
        Возвращает класс модели по имени.
        
        Args:
            display_name: Отображаемое имя модели
            
        Returns:
            Класс модели или None
        """
        return self._models.get(display_name)
    
    def get_model_instance(self, display_name: str, **kwargs) -> Optional[object]:
        """
        Возвращает экземпляр модели, создавая его при необходимости.
        
        Args:
            display_name: Отображаемое имя модели
            **kwargs: Параметры для конструктора
            
        Returns:
            Экземпляр модели или None
        """
        # Используем кеш экземпляров
        cache_key = f"{display_name}_{str(sorted(kwargs.items()))}"
        
        if cache_key not in self._instances:
            model_class = self.get_model_class(display_name)
            if model_class:
                self._instances[cache_key] = instantiate_model(model_class, **kwargs)
        
        return self._instances.get(cache_key)
    
    def list_models(self) -> List[str]:
        """Возвращает список всех доступных моделей."""
        return list_available_models(self._models)
    
    def get_model_by_organ(self, organ_key: str) -> Optional[Type]:
        """
        Находит модель по ключу органа.
        
        Args:
            organ_key: Ключ органа (например, 'lung', 'liver')
            
        Returns:
            Класс модели или None
        """
        for model_class in self._models.values():
            if getattr(model_class, 'organ_key', None) == organ_key:
                return model_class
        return None