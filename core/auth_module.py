"""
Модуль аутентификации для ролевой модели User/Admin.
Хранит пароль в хешированном виде (SHA256).
"""

import hashlib
import json
import os
from pathlib import Path


class AuthManager:
    """Управление аутентификацией и ролями пользователей"""
    
    def __init__(self, config_path: str = "auth_config.json"):
        self.config_path = Path(config_path)
        self.current_role = "User"  # По умолчанию обычный пользователь
        self._load_or_create_config()
    
    def _load_or_create_config(self):
        """Загружает или создает конфигурацию с хешем пароля"""
        if not self.config_path.exists():
            # Пароль по умолчанию: "admin123"
            default_password = "admin123"
            hashed = self._hash_password(default_password)
            config = {
                "admin_password_hash": hashed,
                "password_hint": "Пароль по умолчанию: admin123"
            }
            self._save_config(config)
            print(f"⚠️ Создан файл аутентификации. Пароль по умолчанию: {default_password}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def _save_config(self, config: dict):
        """Сохраняет конфигурацию"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _hash_password(self, password: str) -> str:
        """Хеширует пароль с использованием SHA256"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def verify_admin_password(self, password: str) -> bool:
        """Проверяет пароль администратора"""
        hashed_input = self._hash_password(password)
        return hashed_input == self.config["admin_password_hash"]
    
    def login_admin(self, password: str) -> bool:
        """Вход в режим администратора"""
        if self.verify_admin_password(password):
            self.current_role = "Admin"
            return True
        return False
    
    def logout(self):
        """Выход из режима администратора"""
        self.current_role = "User"
    
    def is_admin(self) -> bool:
        """Проверяет, является ли текущий пользователь администратором"""
        return self.current_role == "Admin"
    
    def change_admin_password(self, old_password: str, new_password: str) -> bool:
        """Изменяет пароль администратора"""
        if not self.verify_admin_password(old_password):
            return False
        
        self.config["admin_password_hash"] = self._hash_password(new_password)
        self._save_config(self.config)
        return True
