"""
Менеджер состояний пользовательского интерфейса.

Централизованное управление состоянием UI виджетов для предотвращения
некорректных действий пользователя (например, нажатие "Сегментировать" во время загрузки).
"""

from typing import Dict, Any, Optional
from enum import Enum


class UIState(Enum):
    """Возможные состояния приложения."""
    INITIAL = "initial"
    VOLUME_LOADED = "volume_loaded"
    ROI1_DEFINED = "roi1_defined"  # ROI 1 определены
    ROI2_DEFINED = "roi2_defined"  # ROI 2 определены
    ROI_DEFINED = "roi_defined"
    SEGMENTING = "segmenting"
    MASK_READY = "mask_ready"
    REFINING = "refining"
    SAVING = "saving"


class UIStateManager:
    """
    Централизованный менеджер состояний UI.
    
    Управляет доступностью виджетов в зависимости от текущего состояния приложения.
    """
    
    def __init__(self) -> None:
        """Инициализирует менеджер состояний."""
        self._current_state: UIState = UIState.INITIAL
        self._widgets: Dict[str, Any] = {}
        self._state_callbacks: Dict[UIState, callable] = {}
    
    def register_widgets(self, widgets: Dict[str, Any]) -> None:
        """
        Регистрирует виджеты для управления.
        
        Args:
            widgets: Словарь виджетов {имя: объект_виджета}
            
        Example:
            >>> manager.register_widgets({
            ...     'btn_load': self.btn_load,
            ...     'btn_segment': self.btn_segment,
            ...     'slider_hu_min': self.slider_hu_min
            ... })
        """
        self._widgets = widgets
    
    def register_state_callback(self, state: UIState, callback: callable) -> None:
        """
        Регистрирует callback для вызова при переходе в состояние.
        
        Args:
            state: Состояние для отслеживания
            callback: Функция для вызова
        """
        self._state_callbacks[state] = callback
    
    @property
    def current_state(self) -> UIState:
        """Возвращает текущее состояние."""
        return self._current_state
    
    def transition_to(self, new_state: UIState) -> None:
        """
        Переводит UI в новое состояние.
        
        Args:
            new_state: Целевое состояние
            
        Example:
            >>> manager.transition_to(UIState.VOLUME_LOADED)
        """
        if new_state == self._current_state:
            return
        
        print(f"[StateManager] {self._current_state.value} -> {new_state.value}")
        self._current_state = new_state
        self._apply_state()
        
        # Вызов callback если зарегистрирован
        if new_state in self._state_callbacks:
            self._state_callbacks[new_state]()
    
    def _apply_state(self) -> None:
        """Применяет конфигурацию виджетов для текущего состояния."""
        state = self._current_state
        
        if state == UIState.INITIAL:
            self._set_initial_state()
        elif state == UIState.ROI1_DEFINED:  # ДОБАВЬТЕ ЭТИ ДВЕ СТРОКИ
            self._set_roi1_defined_state()
        elif state == UIState.VOLUME_LOADED:
            self._set_volume_loaded_state()
        elif state == UIState.ROI_DEFINED:
            self._set_roi_defined_state()
        elif state == UIState.SEGMENTING:
            self._set_segmenting_state()
        elif state == UIState.MASK_READY:
            self._set_mask_ready_state()
        elif state == UIState.REFINING:
            self._set_refining_state()
        elif state == UIState.SAVING:
            self._set_saving_state()
    
    def _set_widget_enabled(self, widget_name: str, enabled: bool) -> None:
        """Устанавливает enabled состояние виджета."""
        if widget_name in self._widgets:
            widget = self._widgets[widget_name]
            if hasattr(widget, 'setEnabled'):
                widget.setEnabled(enabled)
    
    def _set_initial_state(self) -> None:
        """Начальное состояние - доступна только загрузка."""
        self._set_widget_enabled('btn_load', True)
        self._set_widget_enabled('btn_draw_roi1', False)
        self._set_widget_enabled('btn_draw_roi2', False)
        self._set_widget_enabled('btn_reset_roi', False)
        self._set_widget_enabled('btn_segment', False)
        self._set_widget_enabled('btn_apply_refinement', False)
        self._set_widget_enabled('btn_reset_mask', False)
        self._set_widget_enabled('btn_save', False)
        self._set_widget_enabled('slice_slider', False)
        
        # Параметры постобработки
        self._set_refinement_params_enabled(False)
    
    def _set_volume_loaded_state(self) -> None:
        """Volume загружен - доступны ROI и навигация."""
        self._set_widget_enabled('btn_load', True)
        self._set_widget_enabled('btn_draw_roi1', True)
        self._set_widget_enabled('btn_draw_roi2', False)
        self._set_widget_enabled('btn_reset_roi', True)
        self._set_widget_enabled('btn_segment', False)
        self._set_widget_enabled('btn_apply_refinement', False)
        self._set_widget_enabled('btn_reset_mask', False)
        self._set_widget_enabled('btn_save', False)
        self._set_widget_enabled('slice_slider', True)
        
        self._set_refinement_params_enabled(False)
    def _set_roi1_defined_state(self) -> None:
        """ROI 1 определен - доступно рисование ROI 2."""
        self._set_widget_enabled('btn_load', True)
        self._set_widget_enabled('btn_draw_roi1', True)
        self._set_widget_enabled('btn_draw_roi2', True)  # ВОТ ОНО!
        self._set_widget_enabled('btn_reset_roi', True)
        self._set_widget_enabled('btn_segment', False)
        self._set_widget_enabled('btn_apply_refinement', False)
        self._set_widget_enabled('btn_reset_mask', False)
        self._set_widget_enabled('btn_save', False)
        self._set_widget_enabled('slice_slider', True)
    
        self._set_refinement_params_enabled(False)
   
    def _set_roi_defined_state(self) -> None:
        """ROI определены - доступна сегментация."""
        self._set_widget_enabled('btn_load', True)
        self._set_widget_enabled('btn_draw_roi1', True)
        self._set_widget_enabled('btn_draw_roi2', True)
        self._set_widget_enabled('btn_reset_roi', True)
        self._set_widget_enabled('btn_segment', True)
        self._set_widget_enabled('btn_apply_refinement', False)
        self._set_widget_enabled('btn_reset_mask', False)
        self._set_widget_enabled('btn_save', False)
        self._set_widget_enabled('slice_slider', True)
        
        self._set_refinement_params_enabled(False)
    
    def _set_segmenting_state(self) -> None:
        """Идет сегментация - заблокированы все действия."""
        self._set_widget_enabled('btn_load', False)
        self._set_widget_enabled('btn_draw_roi1', False)
        self._set_widget_enabled('btn_draw_roi2', False)
        self._set_widget_enabled('btn_reset_roi', False)
        self._set_widget_enabled('btn_segment', False)
        self._set_widget_enabled('btn_apply_refinement', False)
        self._set_widget_enabled('btn_reset_mask', False)
        self._set_widget_enabled('btn_save', False)
        self._set_widget_enabled('slice_slider', True)  # Можно смотреть срезы
        
        self._set_refinement_params_enabled(False)
    
    def _set_mask_ready_state(self) -> None:
        """Маска готова - доступны постобработка и сохранение."""
        self._set_widget_enabled('btn_load', True)
        self._set_widget_enabled('btn_draw_roi1', True)
        self._set_widget_enabled('btn_draw_roi2', True)
        self._set_widget_enabled('btn_reset_roi', True)
        self._set_widget_enabled('btn_segment', True)
        self._set_widget_enabled('btn_apply_refinement', True)
        self._set_widget_enabled('btn_reset_mask', True)
        self._set_widget_enabled('btn_save', True)
        self._set_widget_enabled('slice_slider', True)
        
        self._set_refinement_params_enabled(True)
    
    def _set_refining_state(self) -> None:
        """Идет постобработка - заблокированы действия с маской."""
        self._set_widget_enabled('btn_load', False)
        self._set_widget_enabled('btn_draw_roi1', False)
        self._set_widget_enabled('btn_draw_roi2', False)
        self._set_widget_enabled('btn_reset_roi', False)
        self._set_widget_enabled('btn_segment', False)
        self._set_widget_enabled('btn_apply_refinement', False)
        self._set_widget_enabled('btn_reset_mask', False)
        self._set_widget_enabled('btn_save', False)
        self._set_widget_enabled('slice_slider', True)
        
        self._set_refinement_params_enabled(False)
    
    def _set_saving_state(self) -> None:
        """Идет сохранение."""
        self._set_widget_enabled('btn_load', False)
        self._set_widget_enabled('btn_save', False)
        self._set_widget_enabled('btn_apply_refinement', False)
    
    def _set_refinement_params_enabled(self, enabled: bool) -> None:
        """Устанавливает доступность параметров постобработки."""
        param_widgets = [
            'slider_hu_min', 'slider_hu_max',
            'slider_dilation', 'slider_closing',
            'cb_fill_holes',
            'btn_conservative_preset', 'btn_aggressive_preset'
        ]
        for widget_name in param_widgets:
            self._set_widget_enabled(widget_name, enabled)
    
    def can_load_dicom(self) -> bool:
        """Проверяет, можно ли загрузить DICOM."""
        return self._current_state not in [UIState.SEGMENTING, UIState.REFINING, UIState.SAVING]
    
    def can_segment(self) -> bool:
        """Проверяет, можно ли запустить сегментацию."""
        return self._current_state in [UIState.ROI_DEFINED, UIState.MASK_READY]
    
    def can_refine(self) -> bool:
        """Проверяет, можно ли запустить постобработку."""
        return self._current_state == UIState.MASK_READY
    
    def can_save(self) -> bool:
        """Проверяет, можно ли сохранить маску."""
        return self._current_state == UIState.MASK_READY