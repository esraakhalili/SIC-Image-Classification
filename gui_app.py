#!/usr/bin/env python3

import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
import threading
import time

from PyQt6.QtWidgets import (  # pyright: ignore[reportMissingImports]
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QTextEdit, QProgressBar, QGroupBox, QGridLayout,
    QMessageBox, QCheckBox, QSlider, QSplitter, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize  # pyright: ignore[reportMissingImports]
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor  # pyright: ignore[reportMissingImports]

from src.trainer import ClassificationTrainer
from src.predictor import ClassificationPredictor
from src.exporter import ModelExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingThread(QThread):
    """Thread for running training in background."""
    progress_update = pyqtSignal(int, float, float)  # epoch, train_loss, val_acc
    training_complete = pyqtSignal(str, dict)  # best_model_path, results
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, trainer: ClassificationTrainer, train_params: dict):
        super().__init__()
        self.trainer = trainer
        self.train_params = train_params
        self.is_running = False
    
    def run(self):
        """Run training."""
        try:
            self.is_running = True
            self.log_message.emit("Starting training...")
            
            results = self.trainer.train(**self.train_params)
            
            if self.is_running:
                self.training_complete.emit(
                    results.get('best_model', ''),
                    results
                )
                self.log_message.emit("Training completed successfully!")
        except Exception as e:
            self.error_occurred.emit(str(e))
            logger.error(f"Training error: {e}")
    
    def stop(self):
        """Stop training."""
        self.is_running = False


class PredictionThread(QThread):
    """Thread for running predictions."""
    prediction_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)  # current, total
    
    def __init__(self, predictor: ClassificationPredictor, source: str, is_folder: bool = False):
        super().__init__()
        self.predictor = predictor
        self.source = source
        self.is_folder = is_folder
    
    def run(self):
        """Run prediction."""
        try:
            if self.is_folder:
                results = self.predictor.predict_folder(
                    folder_path=self.source,
                    save=False
                )
                self.prediction_complete.emit({"type": "folder", "results": results})
            else:
                results = self.predictor.predict_image(
                    image_path=self.source,
                    save=False
                )
                self.prediction_complete.emit({"type": "image", "results": results})
        except Exception as e:
            self.error_occurred.emit(str(e))


class ExportThread(QThread):
    """Thread for exporting models."""
    export_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, exporter: ModelExporter, export_params: dict):
        super().__init__()
        self.exporter = exporter
        self.export_params = export_params
    
    def run(self):
        """Run export."""
        try:
            format_type = self.export_params.pop('format', 'onnx')
            
            if format_type == 'onnx':
                results = self.exporter.export_onnx(**self.export_params)
            else:
                results = self.exporter.export(format=format_type, **self.export_params)
            
            self.export_complete.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))


class TrainingTab(QWidget):
    """Training configuration and monitoring tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.training_thread: Optional[TrainingThread] = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Model Selection Group
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov8n-cls.pt",
            "yolov8s-cls.pt",
            "yolov8m-cls.pt",
            "yolov8l-cls.pt",
            "yolov8x-cls.pt"
        ])
        model_layout.addWidget(self.model_combo, 0, 1)
        
        model_layout.addWidget(QLabel("Dataset Path:"), 1, 0)
        dataset_layout = QHBoxLayout()
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Select dataset directory...")
        dataset_btn = QPushButton("Browse...")
        dataset_btn.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(self.dataset_path)
        dataset_layout.addWidget(dataset_btn)
        model_layout.addLayout(dataset_layout, 1, 1)
        
        model_layout.addWidget(QLabel("Device:"), 2, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "0", "1"])
        model_layout.addWidget(self.device_combo, 2, 1)
        
        self.pretrained_check = QCheckBox("Use Pretrained Weights")
        self.pretrained_check.setChecked(True)
        model_layout.addWidget(self.pretrained_check, 3, 0, 1, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training Parameters Group
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        params_layout.addWidget(self.epochs_spin, 0, 1)
        
        params_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        params_layout.addWidget(self.batch_spin, 0, 3)
        
        params_layout.addWidget(QLabel("Image Size:"), 1, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1024)
        self.imgsz_spin.setValue(224)
        self.imgsz_spin.setSingleStep(32)
        params_layout.addWidget(self.imgsz_spin, 1, 1)
        
        params_layout.addWidget(QLabel("Workers:"), 1, 2)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(8)
        params_layout.addWidget(self.workers_spin, 1, 3)
        
        params_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        params_layout.addWidget(self.lr_spin, 2, 1)
        
        params_layout.addWidget(QLabel("Patience:"), 2, 2)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 1000)
        self.patience_spin.setValue(50)
        params_layout.addWidget(self.patience_spin, 2, 3)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Project Settings
        project_group = QGroupBox("Project Settings")
        project_layout = QGridLayout()
        
        project_layout.addWidget(QLabel("Project Name:"), 0, 0)
        self.project_edit = QLineEdit()
        self.project_edit.setText("yolov8_classification")
        project_layout.addWidget(self.project_edit, 0, 1)
        
        project_layout.addWidget(QLabel("Run Name:"), 1, 0)
        self.run_name_edit = QLineEdit()
        self.run_name_edit.setText("run")
        project_layout.addWidget(self.run_name_edit, 1, 1)
        
        project_group.setLayout(project_layout)
        layout.addWidget(project_group)
        
        # Progress and Logging
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to train")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px;")
        progress_layout.addWidget(self.status_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        progress_layout.addWidget(self.log_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Control Buttons
        button_layout = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.train_btn.clicked.connect(self.start_training)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_training)
        
        button_layout.addWidget(self.train_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def browse_dataset(self):
        """Browse for dataset directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory", str(Path.home())
        )
        if directory:
            self.dataset_path.setText(directory)
    
    def start_training(self):
        """Start training process."""
        if not self.dataset_path.text():
            QMessageBox.warning(self, "Error", "Please select a dataset directory!")
            return
        
        dataset_path = Path(self.dataset_path.text())
        if not dataset_path.exists():
            QMessageBox.warning(self, "Error", f"Dataset path does not exist: {dataset_path}")
            return
        
        # Initialize trainer
        trainer = ClassificationTrainer(
            model_name=self.model_combo.currentText(),
            pretrained=self.pretrained_check.isChecked(),
            device=self.device_combo.currentText()
        )
        
        # Prepare training parameters
        train_params = {
            "data_path": str(dataset_path),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "workers": self.workers_spin.value(),
            "patience": self.patience_spin.value(),
            "project": self.project_edit.text(),
            "name": self.run_name_edit.text(),
            "lr0": self.lr_spin.value()
        }
        
        # Start training thread
        self.training_thread = TrainingThread(trainer, train_params)
        self.training_thread.progress_update.connect(self.update_progress)
        self.training_thread.training_complete.connect(self.training_finished)
        self.training_thread.error_occurred.connect(self.training_error)
        self.training_thread.log_message.connect(self.add_log)
        
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training in progress...")
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        self.training_thread.start()
    
    def stop_training(self):
        """Stop training process."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.training_thread.terminate()
            self.training_thread.wait()
        
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Training stopped")
        self.add_log("Training stopped by user")
    
    def update_progress(self, epoch: int, train_loss: float, val_acc: float):
        """Update training progress."""
        progress = int((epoch / self.epochs_spin.value()) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(
            f"Epoch {epoch}/{self.epochs_spin.value()} | "
            f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
    
    def training_finished(self, best_model: str, results: dict):
        """Handle training completion."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Training completed! Best model: {best_model}")
        self.add_log(f"Training completed successfully!")
        self.add_log(f"Best model saved at: {best_model}")
        
        QMessageBox.information(
            self, "Training Complete",
            f"Training completed successfully!\n\nBest model: {best_model}"
        )
    
    def training_error(self, error_msg: str):
        """Handle training error."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Training failed!")
        self.add_log(f"ERROR: {error_msg}")
        
        QMessageBox.critical(self, "Training Error", f"Training failed:\n{error_msg}")
    
    def add_log(self, message: str):
        """Add log message."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


class InferenceTab(QWidget):
    """Inference and prediction tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor: Optional[ClassificationPredictor] = None
        self.prediction_thread: Optional[PredictionThread] = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Model Selection
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        
        model_layout.addWidget(QLabel("Model Path:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select trained model file...")
        model_browse_btn = QPushButton("Browse...")
        model_browse_btn.clicked.connect(self.browse_model)
        
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(model_browse_btn)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Source Selection
        source_group = QGroupBox("Input Source")
        source_layout = QVBoxLayout()
        
        source_type_layout = QHBoxLayout()
        source_type_layout.addWidget(QLabel("Source Type:"))
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["Image", "Folder", "Video", "Webcam"])
        self.source_type_combo.currentTextChanged.connect(self.on_source_type_changed)
        source_type_layout.addWidget(self.source_type_combo)
        source_type_layout.addStretch()
        source_layout.addLayout(source_type_layout)
        
        source_path_layout = QHBoxLayout()
        source_path_layout.addWidget(QLabel("Source:"))
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setPlaceholderText("Select source...")
        self.source_browse_btn = QPushButton("Browse...")
        self.source_browse_btn.clicked.connect(self.browse_source)
        source_path_layout.addWidget(self.source_path_edit)
        source_path_layout.addWidget(self.source_browse_btn)
        source_layout.addLayout(source_path_layout)
        
        self.conf_threshold_label = QLabel("Confidence Threshold:")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_value_label = QLabel("0.25")
        self.conf_value_label.setStyleSheet("color: #4CAF50; font-weight: bold; min-width: 50px;")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value_label.setText(f"{v/100:.2f}")
        )
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.conf_threshold_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        source_layout.addLayout(conf_layout)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Results Display
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Image Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("No image selected")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.preview_label.setStyleSheet("""
            border: 2px solid #555;
            background-color: #2b2b2b;
            color: #888;
            font-size: 14px;
        """)
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        
        # Results
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        results_splitter.addWidget(preview_group)
        results_splitter.addWidget(results_group)
        results_splitter.setStretchFactor(0, 1)
        results_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(results_splitter)
        
        # Control Buttons
        button_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.predict_btn.clicked.connect(self.run_prediction)
        
        self.save_check = QCheckBox("Save Predictions")
        self.save_check.setChecked(False)
        
        button_layout.addWidget(self.predict_btn)
        button_layout.addWidget(self.save_check)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.on_source_type_changed("Image")
    
    def on_source_type_changed(self, source_type: str):
        """Handle source type change."""
        if source_type == "Webcam":
            self.source_path_edit.setPlaceholderText("Enter webcam index (0 for default)")
            self.source_browse_btn.setEnabled(False)
        else:
            self.source_browse_btn.setEnabled(True)
            if source_type == "Image":
                self.source_path_edit.setPlaceholderText("Select image file...")
            elif source_type == "Folder":
                self.source_path_edit.setPlaceholderText("Select folder...")
            elif source_type == "Video":
                self.source_path_edit.setPlaceholderText("Select video file...")
    
    def browse_model(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", str(Path.home()),
            "PyTorch Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def browse_source(self):
        """Browse for source file/folder."""
        source_type = self.source_type_combo.currentText()
        
        if source_type == "Image":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", str(Path.home()),
                "Images (*.jpg *.jpeg *.png *.bmp *.ppm);;All Files (*)"
            )
            if file_path:
                self.source_path_edit.setText(file_path)
                self.load_preview_image(file_path)
        
        elif source_type == "Folder":
            directory = QFileDialog.getExistingDirectory(
                self, "Select Folder", str(Path.home())
            )
            if directory:
                self.source_path_edit.setText(directory)
        
        elif source_type == "Video":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", str(Path.home()),
                "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
            )
            if file_path:
                self.source_path_edit.setText(file_path)
    
    def load_preview_image(self, image_path: str):
        """Load image for preview."""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
    
    def run_prediction(self):
        """Run prediction."""
        if not self.model_path_edit.text():
            QMessageBox.warning(self, "Error", "Please select a model file!")
            return
        
        if not self.source_path_edit.text():
            QMessageBox.warning(self, "Error", "Please select a source!")
            return
        
        try:
            # Initialize predictor
            self.predictor = ClassificationPredictor(
                model_path=self.model_path_edit.text(),
                device="auto",
                conf_threshold=self.conf_slider.value() / 100.0
            )
            
            source_type = self.source_type_combo.currentText()
            source = self.source_path_edit.text()
            
            if source_type == "Image":
                results = self.predictor.predict_image(
                    image_path=source,
                    save=self.save_check.isChecked()
                )
                self.display_image_results(results)
            
            elif source_type == "Folder":
                results_list = self.predictor.predict_folder(
                    folder_path=source,
                    save=self.save_check.isChecked()
                )
                self.display_folder_results(results_list)
            
            elif source_type == "Video":
                QMessageBox.information(
                    self, "Video Prediction",
                    "Video prediction will be implemented in the next update."
                )
            
            elif source_type == "Webcam":
                QMessageBox.information(
                    self, "Webcam Prediction",
                    "Webcam prediction will be implemented in the next update."
                )
        
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Prediction failed:\n{str(e)}")
            logger.error(f"Prediction error: {e}")
    
    def display_image_results(self, results: dict):
        """Display image prediction results."""
        self.results_text.clear()
        self.results_text.append("=" * 60)
        self.results_text.append("PREDICTION RESULTS")
        self.results_text.append("=" * 60)
        self.results_text.append(f"\nTop Prediction:")
        self.results_text.append(f"  Class: {results['top1_class']}")
        self.results_text.append(f"  Confidence: {results['top1_confidence']:.4f} ({results['top1_confidence']*100:.2f}%)")
        
        self.results_text.append(f"\nTop 5 Predictions:")
        for i, (cls, conf) in enumerate(zip(results['top5_classes'], results['top5_confidences']), 1):
            self.results_text.append(f"  {i}. {cls}: {conf:.4f} ({conf*100:.2f}%)")
        
        # Load and display image with prediction
        if Path(results['image_path']).exists():
            self.load_preview_image(results['image_path'])
    
    def display_folder_results(self, results_list: List[dict]):
        """Display folder prediction results."""
        self.results_text.clear()
        self.results_text.append("=" * 60)
        self.results_text.append(f"FOLDER PREDICTION RESULTS ({len(results_list)} images)")
        self.results_text.append("=" * 60)
        
        # Count predictions by class
        class_counts = {}
        for result in results_list:
            top_class = result['top1_class']
            class_counts[top_class] = class_counts.get(top_class, 0) + 1
        
        self.results_text.append(f"\nSummary:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results_list)) * 100
            self.results_text.append(f"  {cls}: {count} ({percentage:.1f}%)")
        
        self.results_text.append(f"\nDetailed Results:")
        for i, result in enumerate(results_list[:20], 1):  # Show first 20
            self.results_text.append(
                f"\n{i}. {Path(result['image_path']).name}: "
                f"{result['top1_class']} ({result['top1_confidence']:.2f})"
            )
        
        if len(results_list) > 20:
            self.results_text.append(f"\n... and {len(results_list) - 20} more images")


class ExportTab(QWidget):
    """Model export tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.export_thread: Optional[ExportThread] = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Model Selection
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        
        model_layout.addWidget(QLabel("Model Path:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select model file to export...")
        model_browse_btn = QPushButton("Browse...")
        model_browse_btn.clicked.connect(self.browse_model)
        
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(model_browse_btn)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Export Settings
        settings_group = QGroupBox("Export Settings")
        settings_layout = QGridLayout()
        
        settings_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["onnx", "torchscript", "coreml"])
        settings_layout.addWidget(self.format_combo, 0, 1)
        
        settings_layout.addWidget(QLabel("Output Path:"), 1, 0)
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output location...")
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(output_browse_btn)
        settings_layout.addLayout(output_layout, 1, 1)
        
        settings_layout.addWidget(QLabel("Image Size:"), 2, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1024)
        self.imgsz_spin.setValue(224)
        self.imgsz_spin.setSingleStep(32)
        settings_layout.addWidget(self.imgsz_spin, 2, 1)
        
        self.half_check = QCheckBox("FP16 Quantization (Half Precision)")
        settings_layout.addWidget(self.half_check, 3, 0, 1, 2)
        
        self.simplify_check = QCheckBox("Simplify ONNX Model")
        self.simplify_check.setChecked(True)
        settings_layout.addWidget(self.simplify_check, 4, 0, 1, 2)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Status
        status_group = QGroupBox("Export Status")
        status_layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        status_layout.addWidget(self.status_text)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Export Button
        self.export_btn = QPushButton("Export Model")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.export_btn.clicked.connect(self.export_model)
        
        layout.addWidget(self.export_btn)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def browse_model(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", str(Path.home()),
            "PyTorch Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            # Auto-set output path
            if not self.output_path_edit.text():
                model_path = Path(file_path)
                format_ext = {
                    "onnx": ".onnx",
                    "torchscript": ".pt",
                    "coreml": ".mlmodel"
                }
                ext = format_ext.get(self.format_combo.currentText(), ".onnx")
                output_path = model_path.parent / f"{model_path.stem}{ext}"
                self.output_path_edit.setText(str(output_path))
    
    def browse_output(self):
        """Browse for output location."""
        format_type = self.format_combo.currentText()
        ext_map = {
            "onnx": "ONNX Models (*.onnx)",
            "torchscript": "TorchScript Models (*.pt)",
            "coreml": "CoreML Models (*.mlmodel)"
        }
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Exported Model", str(Path.home()),
            ext_map.get(format_type, "All Files (*)")
        )
        if file_path:
            self.output_path_edit.setText(file_path)
    
    def export_model(self):
        """Export model."""
        if not self.model_path_edit.text():
            QMessageBox.warning(self, "Error", "Please select a model file!")
            return
        
        if not self.output_path_edit.text():
            QMessageBox.warning(self, "Error", "Please specify output path!")
            return
        
        try:
            exporter = ModelExporter(
                model_path=self.model_path_edit.text(),
                device="cpu"
            )
            
            export_params = {
                "format": self.format_combo.currentText(),
                "output_path": self.output_path_edit.text(),
                "imgsz": self.imgsz_spin.value(),
                "half": self.half_check.isChecked(),
                "simplify": self.simplify_check.isChecked() if self.format_combo.currentText() == "onnx" else False,
                "opset": 12
            }
            
            self.status_text.clear()
            self.status_text.append("Starting export...")
            self.export_btn.setEnabled(False)
            
            # Run export in thread
            self.export_thread = ExportThread(exporter, export_params)
            self.export_thread.export_complete.connect(self.export_finished)
            self.export_thread.error_occurred.connect(self.export_error)
            self.export_thread.progress_update.connect(self.status_text.append)
            
            self.export_thread.start()
        
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Export failed:\n{str(e)}")
            logger.error(f"Export error: {e}")
    
    def export_finished(self, results: dict):
        """Handle export completion."""
        self.export_btn.setEnabled(True)
        self.status_text.append("Export completed successfully!")
        self.status_text.append(f"Output: {results.get('output_path', 'N/A')}")
        self.status_text.append(f"Size: {results.get('model_size_mb', 0):.2f} MB")
        
        QMessageBox.information(
            self, "Export Complete",
            f"Model exported successfully!\n\n"
            f"Format: {results.get('format', 'N/A')}\n"
            f"Output: {results.get('output_path', 'N/A')}\n"
            f"Size: {results.get('model_size_mb', 0):.2f} MB"
        )
    
    def export_error(self, error_msg: str):
        """Handle export error."""
        self.export_btn.setEnabled(True)
        self.status_text.append(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Export Error", f"Export failed:\n{error_msg}")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.apply_styles()
    
    def init_ui(self):
        """Initialize main UI."""
        self.setWindowTitle("YOLOv8 Classification Suite - Professional GUI")
        self.setGeometry(100, 100, 1600, 1000)
        
        # High DPI scaling is enabled by default in PyQt6
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Title
        title_label = QLabel("YOLOv8 Classification Suite")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #4CAF50; padding: 10px;")
        layout.addWidget(title_label)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create tabs
        self.training_tab = TrainingTab()
        self.inference_tab = InferenceTab()
        self.export_tab = ExportTab()
        
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.inference_tab, "Inference")
        self.tabs.addTab(self.export_tab, "Export")
        
        layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border-top: 1px solid #555;
            }
            QStatusBar::item {
                border: none;
            }
        """)
        self.statusBar().showMessage("Ready")
    
    def apply_styles(self):
        """Apply professional styling with improved contrast and aesthetics."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #3a3a3a;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #404040;
                color: #b0b0b0;
                padding: 12px 24px;
                margin-right: 3px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border: 1px solid #555;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
                color: #4CAF50;
                font-weight: bold;
                border-bottom: 2px solid #4CAF50;
            }
            QTabBar::tab:hover {
                background-color: #4a4a4a;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #3a3a3a;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #4CAF50;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 8px;
                border: 2px solid #555;
                border-radius: 4px;
                background-color: #2b2b2b;
                color: #e0e0e0;
                selection-background-color: #4CAF50;
                selection-color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #4CAF50;
                background-color: #333333;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #404040;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #b0b0b0;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a;
                border: 2px solid #555;
                selection-background-color: #4CAF50;
                selection-color: #ffffff;
                color: #e0e0e0;
            }
            QTextEdit {
                border: 2px solid #555;
                border-radius: 4px;
                background-color: #2b2b2b;
                color: #e0e0e0;
                font-family: 'Courier New', monospace;
                padding: 8px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #404040;
                color: #e0e0e0;
                border: 2px solid #555;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QPushButton:disabled {
                background-color: #2b2b2b;
                color: #666;
                border-color: #444;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #555;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:hover {
                border-color: #4CAF50;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border-color: #4CAF50;
            }
            QProgressBar {
                border: 2px solid #555;
                border-radius: 4px;
                text-align: center;
                background-color: #2b2b2b;
                color: #e0e0e0;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 8px;
                background: #2b2b2b;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 2px solid #555;
                width: 18px;
                margin: -4px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #66BB6A;
            }
            QScrollBar:vertical {
                border: none;
                background: #2b2b2b;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #555;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #2b2b2b;
                height: 12px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #555;
                min-width: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #666;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QSplitter::handle {
                background-color: #555;
            }
            QSplitter::handle:horizontal {
                width: 3px;
            }
            QSplitter::handle:vertical {
                height: 3px;
            }
            QSplitter::handle:hover {
                background-color: #666;
            }
        """)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # High DPI scaling is enabled by default in PyQt6, no need to set attributes
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

