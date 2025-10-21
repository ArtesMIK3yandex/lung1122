"""
Lung Segmenter GUI - Точка входа приложения.

Модульная архитектура для сегментации медицинских изображений.
Поддерживает множественные модели и органы.
"""

import sys
import os
from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gui.main_window import LungSegmenterGUI


def setup_high_dpi() -> None:
    """Настройка поддержки High DPI дисплеев."""
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


def main() -> int:
    """
    Главная функция запуска приложения.
    
    Returns:
        Код возврата приложения
    """
    # Настройка High DPI
    setup_high_dpi()
    
    # Создание приложения
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Путь к конфигурации
    config_path = project_root / "config.yaml"
    
    # Создание главного окна
    try:
        window = LungSegmenterGUI(config_path=str(config_path))
        window.show()
    except Exception as e:
        print(f"[FATAL ERROR] Не удалось запустить приложение: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Запуск event loop
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())