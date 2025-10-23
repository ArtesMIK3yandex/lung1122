"""
Система логирования действий в JSON-формате.
Группирует действия по сессиям и датам.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import uuid


class ActionLogger:
    """Логгер действий пользователя в JSON-формате"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now().isoformat()
        self.current_role = "User"
        self.actions: List[Dict[str, Any]] = []
    
    def set_role(self, role: str):
        """Устанавливает текущую роль (User/Admin)"""
        self.current_role = role
    
    def log_action(self, action_type: str, **kwargs):
        """
        Логирует действие пользователя
        
        Args:
            action_type: Тип действия (load_dicom, run_segmentation, etc.)
            **kwargs: Дополнительные параметры действия
        """
        action = {
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            **kwargs
        }
        self.actions.append(action)
        self._save_to_file()
    
    def _get_log_file_path(self) -> Path:
        """Возвращает путь к файлу лога за сегодняшний день"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.logs_dir / f"{today}.json"
    
    def _save_to_file(self):
        """Сохраняет текущую сессию в файл"""
        log_file = self._get_log_file_path()
        
        # Загружаем существующие данные
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"sessions": []}
        
        # Ищем текущую сессию или создаем новую
        session_found = False
        for session in data["sessions"]:
            if session.get("session_id") == self.session_id:
                session["actions"] = self.actions
                session_found = True
                break
        
        if not session_found:
            data["sessions"].append({
                "session_id": self.session_id,
                "session_start": self.session_start,
                "user_role": self.current_role,
                "actions": self.actions
            })
        
        # Сохраняем
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def start_new_session(self):
        """Начинает новую сессию логирования"""
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now().isoformat()
        self.actions = []
    
    # Предопределенные методы для частых действий
    
    def log_dicom_load(self, file_path: str, series_uid: str = None):
        """Логирует загрузку DICOM"""
        self.log_action("load_dicom", file=file_path, series_uid=series_uid)
    
    def log_slice_change(self, slice_number: int):
        """Логирует изменение среза"""
        self.log_action("interaction_slice_change", slice=slice_number)
    
    def log_slice_range_set(self, start_slice: int, end_slice: int):
        """Логирует установку рабочего диапазона срезов"""
        self.log_action("set_slice_range", start_slice=start_slice, end_slice=end_slice)
    
    def log_segmentation_run(self, mode: str = None):
        """Логирует запуск сегментации"""
        self.log_action("run_segmentation", mode=mode)
    
    def log_processing_mode(self, mode_name: str, parameters: dict = None):
        """Логирует выбор режима обработки"""
        self.log_action("select_processing_mode", mode=mode_name, params=parameters)
    
    def log_save_result(self, file_path: str, format_type: str = "nifti"):
        """Логирует сохранение результата"""
        self.log_action("save_result", file=file_path, format=format_type)
    
    def log_admin_action(self, admin_action: str, **kwargs):
        """Логирует действие администратора"""
        self.log_action("admin_action", admin_action=admin_action, **kwargs)
