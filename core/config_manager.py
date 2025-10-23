"""
Менеджер конфигурации для управления режимами обработки,
видимостью модулей и компоновкой интерфейса.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


class ConfigManager:
    """Управление глобальной конфигурацией приложения"""
    
    DEFAULT_CONFIG = {
        "processing_modes": [
            {
                "id": "mode_1",
                "name": "Базовая сегментация лёгких",
                "parameters": {
                    "threshold": -500,
                    "smooth": True
                }
            },
            {
                "id": "mode_2",
                "name": "Детальная сегментация",
                "parameters": {
                    "threshold": -600,
                    "smooth": True,
                    "refine": True
                }
            }
        ],
        "modules": {
            "segmentation": {
                "visible": True,
                "enabled": True,
                "order": 1,
                "name": "Сегментация",
                "removable": False  # Неотключаемый модуль
            },
            "statistics": {
                "visible": True,
                "enabled": True,
                "order": 2,
                "name": "Статистика по исследованию",
                "removable": True
            }
        },
        "ui_layout": {
            "projection_order": ["axial", "sagittal", "coronal"],
            "toolbar_order": ["load", "save", "reset"]
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> dict:
        """Загружает или создает конфигурацию по умолчанию"""
        if not self.config_path.exists():
            self._save_config(self.DEFAULT_CONFIG)
            print(f"✓ Создан файл конфигурации: {self.config_path}")
            return self.DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки конфигурации: {e}")
            return self.DEFAULT_CONFIG.copy()
    
    def _save_config(self, config: dict = None):
        """Сохраняет конфигурацию в файл"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения конфигурации: {e}")
    
    def save(self):
        """Сохраняет текущую конфигурацию"""
        self._save_config()
    
    # === УПРАВЛЕНИЕ РЕЖИМАМИ ОБРАБОТКИ (CRUD) ===
    
    def get_processing_modes(self) -> List[Dict[str, Any]]:
        """Возвращает список режимов обработки"""
        return self.config.get("processing_modes", [])
    
    def add_processing_mode(self, name: str, parameters: dict) -> str:
        """Добавляет новый режим обработки"""
        import uuid
        mode_id = f"mode_{uuid.uuid4().hex[:8]}"
        
        new_mode = {
            "id": mode_id,
            "name": name,
            "parameters": parameters
        }
        
        self.config["processing_modes"].append(new_mode)
        self.save()
        return mode_id
    
    def update_processing_mode(self, mode_id: str, name: str = None, parameters: dict = None):
        """Обновляет существующий режим"""
        for mode in self.config["processing_modes"]:
            if mode["id"] == mode_id:
                if name:
                    mode["name"] = name
                if parameters:
                    mode["parameters"] = parameters
                self.save()
                return True
        return False
    
    def delete_processing_mode(self, mode_id: str) -> bool:
        """Удаляет режим обработки"""
        modes = self.config["processing_modes"]
        original_count = len(modes)
        self.config["processing_modes"] = [m for m in modes if m["id"] != mode_id]
        
        if len(self.config["processing_modes"]) < original_count:
            self.save()
            return True
        return False
    
    # === УПРАВЛЕНИЕ МОДУЛЯМИ ===
    
    def get_modules(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает конфигурацию модулей"""
        return self.config.get("modules", {})
    
    def get_visible_modules(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает только видимые модули"""
        return {
            k: v for k, v in self.config.get("modules", {}).items()
            if v.get("visible", True)
        }
    
    def set_module_visibility(self, module_id: str, visible: bool):
        """Устанавливает видимость модуля"""
        if module_id in self.config["modules"]:
            # Проверка на неотключаемые модули
            if not self.config["modules"][module_id].get("removable", True):
                print(f"⚠️ Модуль '{module_id}' не может быть скрыт")
                return False
            
            self.config["modules"][module_id]["visible"] = visible
            self.save()
            return True
        return False
    
    def update_module_name(self, module_id: str, new_name: str):
        """Обновляет отображаемое имя модуля"""
        if module_id in self.config["modules"]:
            self.config["modules"][module_id]["name"] = new_name
            self.save()
            return True
        return False
    
    def register_module(self, module_id: str, config: Dict[str, Any]):
        """Регистрирует новый модуль в конфигурации"""
        if module_id not in self.config["modules"]:
            self.config["modules"][module_id] = {
                "visible": config.get("visible", True),
                "enabled": config.get("enabled", True),
                "order": config.get("order", 999),
                "name": config.get("name", module_id),
                "removable": config.get("removable", True)
            }
            self.save()
    
    # === УПРАВЛЕНИЕ КОМПОНОВКОЙ ИНТЕРФЕЙСА ===
    
    def get_ui_layout(self) -> Dict[str, List[str]]:
        """Возвращает настройки компоновки UI"""
        return self.config.get("ui_layout", {})
    
    def set_toolbar_order(self, order: List[str]):
        """Устанавливает порядок кнопок в панели инструментов"""
        if "ui_layout" not in self.config:
            self.config["ui_layout"] = {}
        self.config["ui_layout"]["toolbar_order"] = order
        self.save()
    
    def set_projection_order(self, order: List[str]):
        """Устанавливает порядок окон проекций"""
        if "ui_layout" not in self.config:
            self.config["ui_layout"] = {}
        self.config["ui_layout"]["projection_order"] = order
        self.save()
