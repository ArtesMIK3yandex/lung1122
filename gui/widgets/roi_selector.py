"""
Интерактивный виджет для выбора Region of Interest (ROI).

Позволяет пользователю рисовать прямоугольные области на изображении.
"""

from typing import Tuple, Optional, Callable
from matplotlib.patches import Rectangle


class ROISelector:
    """
    Интерактивный селектор ROI для matplotlib.
    
    Позволяет пользователю нарисовать прямоугольник на изображении
    путем клика и перетаскивания мыши.
    """
    
    def __init__(self, ax, callback: Callable[[Tuple[int, int, int, int]], None]):
        """
        Инициализирует селектор ROI.
        
        Args:
            ax: Matplotlib axis для рисования
            callback: Функция, вызываемая при завершении выбора.
                     Принимает кортеж (x_min, x_max, y_min, y_max)
        """
        self.ax = ax
        self.callback = callback
        self.rect: Optional[Rectangle] = None
        self.x0: Optional[float] = None
        self.y0: Optional[float] = None
        self.press: Optional[Tuple[float, float]] = None
        
        # Connection IDs для отключения
        self.cidpress: Optional[int] = None
        self.cidrelease: Optional[int] = None
        self.cidmotion: Optional[int] = None
    
    def connect(self) -> None:
        """Подключает обработчики событий мыши."""
        canvas = self.ax.figure.canvas
        self.cidpress = canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = canvas.mpl_connect('motion_notify_event', self.on_motion)
    
    def disconnect(self) -> None:
        """Отключает обработчики событий мыши."""
        canvas = self.ax.figure.canvas
        if self.cidpress is not None:
            canvas.mpl_disconnect(self.cidpress)
        if self.cidrelease is not None:
            canvas.mpl_disconnect(self.cidrelease)
        if self.cidmotion is not None:
            canvas.mpl_disconnect(self.cidmotion)
    
    def on_press(self, event) -> None:
        """
        Обработчик нажатия кнопки мыши.
        
        Args:
            event: Событие matplotlib
        """
        if event.inaxes != self.ax:
            return
        
        self.press = (event.xdata, event.ydata)
        self.x0, self.y0 = event.xdata, event.ydata
        
        # Удаляем предыдущий прямоугольник если есть
        if self.rect is not None:
            self.rect.remove()
            self.rect = None
    
    def on_motion(self, event) -> None:
        """
        Обработчик движения мыши.
        
        Args:
            event: Событие matplotlib
        """
        if self.press is None or event.inaxes != self.ax:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        x1, y1 = event.xdata, event.ydata
        
        # Удаляем предыдущий прямоугольник
        if self.rect is not None:
            self.rect.remove()
        
        # Рисуем новый прямоугольник
        width = x1 - self.x0
        height = y1 - self.y0
        self.rect = Rectangle(
            (self.x0, self.y0), width, height,
            linewidth=2, edgecolor='red', facecolor='none',
            linestyle='--', alpha=0.8
        )
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.draw_idle()
    
    def on_release(self, event) -> None:
        """
        Обработчик отпускания кнопки мыши.
        
        Args:
            event: Событие matplotlib
        """
        if self.press is None:
            return
        
        if event.xdata is None or event.ydata is None:
            self.press = None
            return
        
        x1, y1 = event.xdata, event.ydata
        self.press = None
        
        # Вычисляем границы ROI
        x_min = int(min(self.x0, x1))
        x_max = int(max(self.x0, x1))
        y_min = int(min(self.y0, y1))
        y_max = int(max(self.y0, y1))
        
        # Проверяем что ROI не пустой
        if x_max - x_min < 5 or y_max - y_min < 5:
            print("[WARNING] ROI слишком маленький, игнорируется")
            if self.rect is not None:
                self.rect.remove()
                self.rect = None
            self.ax.figure.canvas.draw_idle()
            return
        
        # Делаем финальный прямоугольник более заметным
        if self.rect is not None:
            self.rect.set_linestyle('-')
            self.rect.set_linewidth(2.5)
            self.rect.set_alpha(1.0)
            self.ax.figure.canvas.draw_idle()
        
        # Вызываем callback
        self.callback((x_min, x_max, y_min, y_max))
        
        # Отключаем селектор
        self.disconnect()
    
    def cancel(self) -> None:
        """Отменяет текущий выбор ROI."""
        if self.rect is not None:
            self.rect.remove()
            self.rect = None
        self.disconnect()
        self.ax.figure.canvas.draw_idle()


class ROIManager:
    """
    Менеджер для управления несколькими ROI.
    
    Хранит и управляет ROI для различных срезов и проекций.
    """
    
    def __init__(self):
        """Инициализирует менеджер ROI."""
        self.roi1_3d: Optional[Tuple[int, int, int, int, int]] = None
        self.roi2_3d: Optional[Tuple[int, int, int, int, int]] = None
    
    def set_roi1(self, z_slice: int, x_min: int, x_max: int, y_min: int, y_max: int) -> None:
        """
        Устанавливает первый ROI.
        
        Args:
            z_slice: Номер среза
            x_min, x_max: Границы по X
            y_min, y_max: Границы по Y
        """
        self.roi1_3d = (z_slice, y_min, y_max, x_min, x_max)
    
    def set_roi2(self, z_slice: int, x_min: int, x_max: int, y_min: int, y_max: int) -> None:
        """
        Устанавливает второй ROI.
        
        Args:
            z_slice: Номер среза
            x_min, x_max: Границы по X
            y_min, y_max: Границы по Y
        """
        self.roi2_3d = (z_slice, y_min, y_max, x_min, x_max)
    
    def reset(self) -> None:
        """Сбрасывает все ROI."""
        self.roi1_3d = None
        self.roi2_3d = None
    
    def has_both_rois(self) -> bool:
        """Проверяет, заданы ли оба ROI."""
        return self.roi1_3d is not None and self.roi2_3d is not None
    
    def get_combined_roi_coords(self, volume_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
        """
        Вычисляет 3D координаты ROI, объединяя ROI1 и ROI2.
        
        Args:
            volume_shape: Размеры volume (Z, Y, X)
            
        Returns:
            Кортеж (z0, z1, y0, y1, x0, x1)
            
        Raises:
            ValueError: Если ROI не заданы
        """
        if not self.has_both_rois():
            raise ValueError("Оба ROI должны быть заданы")
        
        z1_slice, y1_0, y1_1, x1_0, x1_1 = self.roi1_3d
        z2_slice, y2_0, y2_1, x2_0, x2_1 = self.roi2_3d
        
        # Объединяем границы
        x0 = min(x1_0, x2_0)
        x1 = max(x1_1, x2_1)
        y0 = min(y1_0, y2_0)
        y1 = max(y1_1, y2_1)
        z0 = min(z1_slice, z2_slice)
        z1 = max(z1_slice, z2_slice)
        
        # Обрезаем по границам volume
        z0 = max(0, z0)
        z1 = min(volume_shape[0] - 1, z1)
        y0 = max(0, y0)
        y1 = min(volume_shape[1] - 1, y1)
        x0 = max(0, x0)
        x1 = min(volume_shape[2] - 1, x1)
        
        return z0, z1, y0, y1, x0, x1
    
    def get_info_text(self) -> str:
        """
        Возвращает текстовое описание ROI для UI.
        
        Returns:
            Строка с информацией о ROI
        """
        if self.roi1_3d is None and self.roi2_3d is None:
            return "ROI не заданы"
        
        lines = []
        if self.roi1_3d is not None:
            z, y0, y1, x0, x1 = self.roi1_3d
            lines.append(f"ROI1: z={z}, x=[{x0}:{x1}], y=[{y0}:{y1}]")
        if self.roi2_3d is not None:
            z, y0, y1, x0, x1 = self.roi2_3d
            lines.append(f"ROI2: z={z}, x=[{x0}:{x1}], y=[{y0}:{y1}]")
        
        return "\n".join(lines)