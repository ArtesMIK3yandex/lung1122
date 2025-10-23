"""
Динамический загрузчик модулей из директории modules/
Автоматически обнаруживает классы, наследующие BaseModule
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, List
import sys


class PluginLoader:
    """Загрузчик плагинов/модулей"""
    
    def __init__(self, modules_dir: str = "modules"):
        self.modules_dir = Path(modules_dir)
        self.loaded_modules: Dict[str, Type] = {}
    
    def discover_modules(self) -> Dict[str, Type]:
        """
        Обнаруживает все модули в директории modules/
        
        Returns:
            Словарь {module_id: ModuleClass}
        """
        self.loaded_modules.clear()
        
        if not self.modules_dir.exists():
            print(f"⚠️ Директория модулей не найдена: {self.modules_dir}")
            return self.loaded_modules
        
        # Добавляем директорию в sys.path для импорта
        modules_path = str(self.modules_dir.parent.absolute())
        if modules_path not in sys.path:
            sys.path.insert(0, modules_path)
        
        # Сканируем все .py файлы
        for file_path in self.modules_dir.glob("*_module.py"):
            if file_path.name.startswith("_"):
                continue
            
            try:
                self._load_module_from_file(file_path)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модуля {file_path}: {e}")
        
        print(f"✓ Загружено модулей: {len(self.loaded_modules)}")
        return self.loaded_modules
    
    def _load_module_from_file(self, file_path: Path):
        """Загружает модуль из файла"""
        module_name = f"modules.{file_path.stem}"
        
        try:
            # Импортируем модуль
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Ищем классы, наследующие BaseModule
            from modules.base_module import BaseModule
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseModule) and obj is not BaseModule:
                    # Создаем временный экземпляр для получения метаданных
                    try:
                        temp_instance = obj()
                        info = temp_instance.get_module_info()
                        module_id = info.get('id', name.lower())
                        
                        self.loaded_modules[module_id] = obj
                        print(f"  ✓ Загружен: {info.get('name', name)}")
                        
                    except Exception as e:
                        print(f"  ⚠️ Ошибка инициализации {name}: {e}")
        
        except Exception as e:
            print(f"⚠️ Ошибка импорта {file_path}: {e}")
    
    def get_module_class(self, module_id: str) -> Type:
        """Возвращает класс модуля по ID"""
        return self.loaded_modules.get(module_id)
    
    def get_available_modules(self) -> List[Dict]:
        """
        Возвращает список метаданных всех доступных модулей
        
        Returns:
            [{'id': '...', 'name': '...', 'description': '...', ...}, ...]
        """
        modules_info = []
        
        for module_id, module_class in self.loaded_modules.items():
            try:
                temp_instance = module_class()
                info = temp_instance.get_module_info()
                modules_info.append(info)
            except:
                continue
        
        return modules_info
    
    def instantiate_module(self, module_id: str, parent=None):
        """
        Создает экземпляр модуля
        
        Args:
            module_id: ID модуля
            parent: Родительский виджет
        
        Returns:
            Экземпляр модуля или None
        """
        module_class = self.get_module_class(module_id)
        
        if module_class:
            try:
                instance = module_class(parent)
                return instance
            except Exception as e:
                print(f"⚠️ Ошибка создания экземпляра {module_id}: {e}")
        
        return None
