"""
Training Module for YOLOv8 Classification
Handles model training with callbacks, logging, and checkpointing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from ultralytics import YOLO
from ultralytics.utils import callbacks

logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """
    Trainer class for YOLOv8 classification models.
    
    This class encapsulates the training logic with proper error handling,
    device management, and callback integration.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n-cls.pt",
        pretrained: bool = True,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the ClassificationTrainer.
        
        Args:
            model_name: Name of the YOLOv8 classification model (e.g., 'yolov8n-cls.pt')
            pretrained: Whether to use pretrained weights
            device: Device to use ('auto', 'cpu', 'cuda', or specific GPU like '0')
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = self._setup_device(device)
        self.model: Optional[YOLO] = None
        
        logger.info(f"Initialized trainer with model: {model_name}")
        logger.info(f"Device: {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """
        Automatically detect and configure the device (CUDA/CPU).
        
        Args:
            device: Device specification or None for auto-detection
            
        Returns:
            Device string for YOLOv8
        """
        if device is None or device.lower() == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("CUDA not available. Using CPU.")
        else:
            device = device.lower()
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
        
        return device
    
    def load_model(self) -> YOLO:
        """
        Load the YOLOv8 classification model.
        
        Returns:
            Loaded YOLO model instance
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = YOLO(self.model_name)
            logger.info("Model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train(
        self,
        data_path: Union[str, Path],
        epochs: int = 100,
        imgsz: int = 224,
        batch: int = 16,
        workers: int = 8,
        patience: int = 50,
        save_period: int = 10,
        project: str = "yolov8_classification",
        name: str = "run",
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Train the YOLOv8 classification model.
        
        Args:
            data_path: Path to the dataset directory
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            workers: Number of worker threads for data loading
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            project: Project name for logging
            name: Run name for logging
            save_dir: Directory to save training outputs
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training results
        """
        if self.model is None:
            self.load_model()
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise ValueError(f"Dataset path does not exist: {data_path}")
        
        logger.info(f"Starting training on dataset: {data_path}")
        logger.info(f"Training parameters: epochs={epochs}, batch={batch}, imgsz={imgsz}")
        
        # Setup callbacks
        callbacks_dict = {}
        
        # Training arguments
        train_args = {
            "data": str(data_path),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "device": self.device,
            "patience": patience,
            "save_period": save_period,
            "project": project,
            "name": name,
            "exist_ok": True,
            "pretrained": self.pretrained,
            **kwargs
        }
        
        if save_dir:
            train_args["project"] = str(Path(save_dir).parent)
            train_args["name"] = str(Path(save_dir).name)
        
        try:
            # Train the model
            results = self.model.train(**train_args)
            
            logger.info("Training completed successfully")
            logger.info(f"Best model saved at: {self.model.trainer.best}")
            
            return {
                "success": True,
                "best_model": str(self.model.trainer.best),
                "results": results,
                "metrics": {
                    "top1_acc": getattr(results, "top1", None),
                    "top5_acc": getattr(results, "top5", None),
                } if hasattr(results, "top1") else {}
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def resume_training(
        self,
        checkpoint_path: Union[str, Path],
        epochs: Optional[int] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
            epochs: Additional epochs to train (None to continue from checkpoint)
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training results
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        try:
            self.model = YOLO(str(checkpoint_path))
            
            # Get original training arguments if available
            train_args = {
                "resume": True,
                "device": self.device,
                **kwargs
            }
            
            if epochs is not None:
                train_args["epochs"] = epochs
            
            results = self.model.train(**train_args)
            
            logger.info("Resumed training completed successfully")
            
            return {
                "success": True,
                "best_model": str(self.model.trainer.best),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to resume training: {e}")
            raise

