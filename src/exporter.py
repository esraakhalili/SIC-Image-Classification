"""
Export Module for YOLOv8 Classification
Handles model export to various formats (ONNX, TorchScript, etc.).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Exporter class for YOLOv8 classification models.
    
    Supports exporting to ONNX, TorchScript, CoreML, and other formats.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the ModelExporter.
        
        Args:
            model_path: Path to the trained model file (.pt)
            device: Device to use for export ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = device or "cpu"  # Export typically uses CPU
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Initialized exporter with model: {self.model_path}")
    
    def export_onnx(
        self,
        output_path: Optional[Union[str, Path]] = None,
        imgsz: int = 224,
        half: bool = False,
        simplify: bool = True,
        opset: int = 12,
        dynamic: bool = False
    ) -> Dict[str, any]:
        """
        Export the model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model (None for auto-generated)
            imgsz: Image size for the exported model
            half: Use FP16 quantization
            simplify: Simplify the ONNX model
            opset: ONNX opset version
            dynamic: Use dynamic batch size
            
        Returns:
            Dictionary containing export results
        """
        logger.info("Starting ONNX export...")
        
        try:
            model = YOLO(str(self.model_path))
            
            # Export arguments
            export_args = {
                "format": "onnx",
                "imgsz": imgsz,
                "half": half,
                "simplify": simplify,
                "opset": opset,
                "device": self.device,
                "dynamic": dynamic
            }
            
            if output_path:
                export_args["file"] = str(output_path)
            
            # Perform export
            exported_path = model.export(**export_args)
            exported_path = Path(exported_path)
            
            logger.info(f"ONNX export completed successfully: {exported_path}")
            
            # Get model info
            model_size = exported_path.stat().st_size / (1024 * 1024)  # MB
            
            return {
                "success": True,
                "format": "onnx",
                "output_path": str(exported_path),
                "model_size_mb": round(model_size, 2),
                "image_size": imgsz,
                "half_precision": half,
                "simplified": simplify,
                "opset_version": opset
            }
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def export_torchscript(
        self,
        output_path: Optional[Union[str, Path]] = None,
        imgsz: int = 224,
        half: bool = False
    ) -> Dict[str, any]:
        """
        Export the model to TorchScript format.
        
        Args:
            output_path: Path to save the TorchScript model (None for auto-generated)
            imgsz: Image size for the exported model
            half: Use FP16 quantization
            
        Returns:
            Dictionary containing export results
        """
        logger.info("Starting TorchScript export...")
        
        try:
            model = YOLO(str(self.model_path))
            
            export_args = {
                "format": "torchscript",
                "imgsz": imgsz,
                "half": half,
                "device": self.device
            }
            
            if output_path:
                export_args["file"] = str(output_path)
            
            exported_path = model.export(**export_args)
            exported_path = Path(exported_path)
            
            logger.info(f"TorchScript export completed successfully: {exported_path}")
            
            model_size = exported_path.stat().st_size / (1024 * 1024)  # MB
            
            return {
                "success": True,
                "format": "torchscript",
                "output_path": str(exported_path),
                "model_size_mb": round(model_size, 2),
                "image_size": imgsz,
                "half_precision": half
            }
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
    
    def export_coreml(
        self,
        output_path: Optional[Union[str, Path]] = None,
        imgsz: int = 224,
        half: bool = False
    ) -> Dict[str, any]:
        """
        Export the model to CoreML format (for Apple devices).
        
        Args:
            output_path: Path to save the CoreML model (None for auto-generated)
            imgsz: Image size for the exported model
            half: Use FP16 quantization
            
        Returns:
            Dictionary containing export results
        """
        logger.info("Starting CoreML export...")
        
        try:
            model = YOLO(str(self.model_path))
            
            export_args = {
                "format": "coreml",
                "imgsz": imgsz,
                "half": half,
                "device": self.device
            }
            
            if output_path:
                export_args["file"] = str(output_path)
            
            exported_path = model.export(**export_args)
            exported_path = Path(exported_path)
            
            logger.info(f"CoreML export completed successfully: {exported_path}")
            
            model_size = exported_path.stat().st_size / (1024 * 1024)  # MB
            
            return {
                "success": True,
                "format": "coreml",
                "output_path": str(exported_path),
                "model_size_mb": round(model_size, 2),
                "image_size": imgsz,
                "half_precision": half
            }
            
        except Exception as e:
            logger.error(f"CoreML export failed: {e}")
            raise
    
    def export(
        self,
        format: str = "onnx",
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Generic export method that supports multiple formats.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
            output_path: Path to save the exported model
            **kwargs: Additional format-specific arguments
            
        Returns:
            Dictionary containing export results
        """
        format = format.lower()
        
        if format == "onnx":
            return self.export_onnx(output_path=output_path, **kwargs)
        elif format == "torchscript":
            return self.export_torchscript(output_path=output_path, **kwargs)
        elif format == "coreml":
            return self.export_coreml(output_path=output_path, **kwargs)
        else:
            # Use YOLOv8's generic export
            logger.info(f"Exporting to format: {format}")
            
            try:
                model = YOLO(str(self.model_path))
                
                export_args = {
                    "format": format,
                    "device": self.device,
                    **kwargs
                }
                
                if output_path:
                    export_args["file"] = str(output_path)
                
                exported_path = model.export(**export_args)
                exported_path = Path(exported_path)
                
                logger.info(f"Export completed successfully: {exported_path}")
                
                model_size = exported_path.stat().st_size / (1024 * 1024)  # MB
                
                return {
                    "success": True,
                    "format": format,
                    "output_path": str(exported_path),
                    "model_size_mb": round(model_size, 2)
                }
                
            except Exception as e:
                logger.error(f"Export failed: {e}")
                raise

