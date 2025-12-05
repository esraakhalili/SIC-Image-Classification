"""
Inference Module for YOLOv8 Classification
Handles prediction on single images, image folders, and video streams.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ClassificationPredictor:
    """
    Predictor class for YOLOv8 classification models.
    
    Supports prediction on single images, batches of images, folders, and video streams.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        conf_threshold: float = 0.25
    ) -> None:
        """
        Initialize the ClassificationPredictor.
        
        Args:
            model_path: Path to the trained model file (.pt)
            device: Device to use ('auto', 'cpu', 'cuda', or specific GPU like '0')
            conf_threshold: Confidence threshold for predictions
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.device = self._setup_device(device)
        self.model: Optional[YOLO] = None
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Initialized predictor with model: {self.model_path}")
        logger.info(f"Device: {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """
        Automatically detect and configure the device (CUDA/CPU).
        
        Args:
            device: Device specification or None for auto-detection
            
        Returns:
            Device string for YOLOv8
        """
        import torch
        
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
        Load the trained YOLOv8 classification model.
        
        Returns:
            Loaded YOLO model instance
        """
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            logger.info("Model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_image(
        self,
        image_path: Union[str, Path],
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, any]:
        """
        Predict on a single image.
        
        Args:
            image_path: Path to the image file
            save: Whether to save the prediction visualization
            save_dir: Directory to save predictions (if save=True)
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            self.load_model()
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Predicting on image: {image_path}")
        
        try:
            results = self.model(str(image_path), conf=self.conf_threshold)
            result = results[0]
            
            # Extract prediction information
            probs = result.probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            top5_indices = probs.top5.tolist()
            top5_confs = probs.top5conf.tolist()
            
            # Get class names
            class_names = result.names
            
            prediction = {
                "image_path": str(image_path),
                "top1_class": class_names[top1_idx],
                "top1_confidence": top1_conf,
                "top1_index": top1_idx,
                "top5_classes": [class_names[idx] for idx in top5_indices],
                "top5_confidences": [float(conf) for conf in top5_confs],
                "top5_indices": top5_indices,
                "all_classes": list(class_names.values()),
                "all_probabilities": probs.data.cpu().numpy().tolist()
            }
            
            if save:
                save_path = self._save_prediction(result, image_path, save_dir)
                prediction["saved_path"] = str(save_path)
            
            logger.info(f"Prediction: {prediction['top1_class']} ({prediction['top1_confidence']:.4f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_folder(
        self,
        folder_path: Union[str, Path],
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, any]]:
        """
        Predict on all images in a folder.
        
        Args:
            folder_path: Path to the folder containing images
            save: Whether to save prediction visualizations
            save_dir: Directory to save predictions (if save=True)
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            self.load_model()
        
        folder_path = Path(folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Folder not found or not a directory: {folder_path}")
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Find all image files
        image_files = [
            f for f in folder_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {folder_path}")
        
        logger.info(f"Found {len(image_files)} images in folder: {folder_path}")
        
        predictions = []
        for image_file in image_files:
            try:
                prediction = self.predict_image(image_file, save=save, save_dir=save_dir)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to predict on {image_file}: {e}")
                continue
        
        logger.info(f"Completed predictions on {len(predictions)} images")
        
        return predictions
    
    def predict_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        fps: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Predict on a video file.
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the output video (optional)
            show: Whether to display the video during processing
            fps: FPS for output video (None to use original)
            
        Returns:
            Dictionary containing video prediction results
        """
        if self.model is None:
            self.load_model()
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Predicting on video: {video_path}")
        
        try:
            # Run prediction
            results = self.model(str(video_path), conf=self.conf_threshold, save=output_path is not None)
            
            if output_path:
                # YOLOv8 saves the video automatically, but we can specify the path
                logger.info(f"Output video saved to: {output_path}")
            
            # Collect frame-by-frame predictions
            frame_predictions = []
            for result in results:
                probs = result.probs
                top1_idx = int(probs.top1)
                top1_conf = float(probs.top1conf)
                class_names = result.names
                
                frame_predictions.append({
                    "top1_class": class_names[top1_idx],
                    "top1_confidence": top1_conf,
                    "top1_index": top1_idx
                })
            
            return {
                "video_path": str(video_path),
                "output_path": str(output_path) if output_path else None,
                "total_frames": len(frame_predictions),
                "frame_predictions": frame_predictions
            }
            
        except Exception as e:
            logger.error(f"Video prediction failed: {e}")
            raise
    
    def predict_stream(
        self,
        source: Union[int, str] = 0,
        show: bool = True,
        save: bool = False,
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Predict on a video stream (webcam or RTSP stream).
        
        Args:
            source: Video source (0 for webcam, or RTSP URL)
            show: Whether to display the stream
            save: Whether to save the stream
            output_path: Path to save the output video (if save=True)
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Starting stream prediction from source: {source}")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if save and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run prediction
                results = self.model(frame_rgb, conf=self.conf_threshold, verbose=False)
                result = results[0]
                
                # Get prediction
                probs = result.probs
                top1_idx = int(probs.top1)
                top1_conf = float(probs.top1conf)
                class_names = result.names
                
                # Draw prediction on frame
                label = f"{class_names[top1_idx]}: {top1_conf:.2f}"
                cv2.putText(
                    frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                
                if show:
                    cv2.imshow('YOLOv8 Classification', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if writer:
                    writer.write(frame)
        
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user")
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
            
            logger.info("Stream prediction completed")
    
    def _save_prediction(
        self,
        result,
        image_path: Path,
        save_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save prediction visualization.
        
        Args:
            result: YOLOv8 result object
            image_path: Original image path
            save_dir: Directory to save predictions
            
        Returns:
            Path to saved prediction
        """
        if save_dir is None:
            save_dir = Path("predictions")
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save using YOLOv8's built-in save method
        save_path = save_dir / f"pred_{image_path.name}"
        result.save(str(save_path))
        
        return save_path

