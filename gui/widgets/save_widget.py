"""
Виджет сохранения результатов в формате NIfTI
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QGroupBox, QFileDialog, QMessageBox)
import nibabel as nib
import numpy as np


class SaveWidget(QWidget):
    """
    Виджет сохранения результатов - НЕОТКЛЮЧАЕМЫЙ.
    Сохранение в формате NIfTI (.nii).
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = None
        self.mask_data = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Настройка UI"""
        layout = QVBoxLayout(self)
        
        # Группа сохранения
        save_group = QGroupBox("Сохранение результата")
        save_layout = QVBoxLayout()
        
        # Информация
        info_label = QLabel("Сохранение маски в формате NIfTI (.nii)")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888;")
        save_layout.addWidget(info_label)
        
        # Кнопка сохранения
        self.save_btn = QPushButton("Сохранить маску")
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setMinimumHeight(40)
        save_layout.addWidget(self.save_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
    
    def set_logger(self, logger):
        """Устанавливает логгер"""
        self.logger = logger
    
    def set_mask_data(self, mask_data: np.ndarray):
        """Устанавливает данные маски для сохранения"""
        self.mask_data = mask_data
        self.save_btn.setEnabled(mask_data is not None)
    
    def _on_save_clicked(self):
        """Обработка сохранения"""
        if self.mask_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return
        
        # Диалог выбора пути
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить маску",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )
        
        if file_path:
            try:
                # Сохранение в NIfTI
                nifti_img = nib.Nifti1Image(self.mask_data, np.eye(4))
                nib.save(nifti_img, file_path)
                
                if self.logger:
                    self.logger.log_save_result(file_path, "nifti")
                
                QMessageBox.information(self, "Успех", 
                                      f"Маска сохранена:\n{file_path}")
            
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", 
                                   f"Не удалось сохранить файл:\n{str(e)}")