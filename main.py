#!/usr/bin/env python3
"""
Main Entry Point for YOLOv8 Classification Suite
Command-line interface for training, inference, and export operations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from src.exporter import ModelExporter
from src.predictor import ClassificationPredictor
from src.trainer import ClassificationTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def train_command(args: argparse.Namespace, config: dict) -> None:
    """Handle training command."""
    logger.info("=" * 60)
    logger.info("YOLOv8 Classification - Training Mode")
    logger.info("=" * 60)
    
    # Get config values with CLI args taking precedence
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})
    
    model_name = args.model or model_config.get("name", "yolov8n-cls.pt")
    pretrained = args.pretrained if args.pretrained is not None else model_config.get("pretrained", True)
    data_path = args.data or dataset_config.get("root", "data")
    device = args.device or training_config.get("device", "auto")
    
    epochs = args.epochs or training_config.get("epochs", 100)
    batch_size = args.batch or training_config.get("batch_size", 16)
    imgsz = args.imgsz or training_config.get("imgsz", 224)
    workers = args.workers or training_config.get("workers", 8)
    patience = args.patience or training_config.get("patience", 50)
    save_period = args.save_period or training_config.get("save_period", 10)
    
    project = args.project or logging_config.get("project", "yolov8_classification")
    name = args.name or logging_config.get("name", "run")
    save_dir = args.save_dir or logging_config.get("save_dir")
    
    # Initialize trainer
    trainer = ClassificationTrainer(
        model_name=model_name,
        pretrained=pretrained,
        device=device
    )
    
    # Start training
    try:
        results = trainer.train(
            data_path=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            workers=workers,
            patience=patience,
            save_period=save_period,
            project=project,
            name=name,
            save_dir=save_dir
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model: {results['best_model']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def predict_command(args: argparse.Namespace) -> None:
    """Handle prediction command."""
    logger.info("=" * 60)
    logger.info("YOLOv8 Classification - Inference Mode")
    logger.info("=" * 60)
    
    if not args.model:
        logger.error("Model path is required for prediction")
        sys.exit(1)
    
    if not args.source:
        logger.error("Source (image/folder/video) is required for prediction")
        sys.exit(1)
    
    device = args.device or "auto"
    conf_threshold = args.conf or 0.25
    
    # Initialize predictor
    predictor = ClassificationPredictor(
        model_path=args.model,
        device=device,
        conf_threshold=conf_threshold
    )
    
    # Check if source is a stream (webcam or RTSP) before converting to Path
    if args.source.isdigit() or args.source.startswith("rtsp://") or args.source.startswith("http://"):
        logger.info("Detected video stream")
        source_input = int(args.source) if args.source.isdigit() else args.source
        predictor.predict_stream(
            source=source_input,
            show=args.show,
            save=args.save,
            output_path=args.output
        )
        return
    
    source = Path(args.source)
    
    try:
        if source.is_file():
            # Check if it's a video file
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
            if source.suffix.lower() in video_extensions:
                logger.info("Detected video file")
                results = predictor.predict_video(
                    video_path=source,
                    output_path=args.output,
                    show=args.show
                )
                logger.info(f"Processed {results['total_frames']} frames")
            else:
                logger.info("Detected image file")
                results = predictor.predict_image(
                    image_path=source,
                    save=args.save,
                    save_dir=args.output
                )
                logger.info(f"Prediction: {results['top1_class']} ({results['top1_confidence']:.4f})")
        
        elif source.is_dir():
            logger.info("Detected image folder")
            results = predictor.predict_folder(
                folder_path=source,
                save=args.save,
                save_dir=args.output
            )
            logger.info(f"Processed {len(results)} images")
        
        else:
            logger.error(f"Invalid source: {args.source}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


def export_command(args: argparse.Namespace, config: dict) -> None:
    """Handle export command."""
    logger.info("=" * 60)
    logger.info("YOLOv8 Classification - Export Mode")
    logger.info("=" * 60)
    
    if not args.model:
        logger.error("Model path is required for export")
        sys.exit(1)
    
    export_config = config.get("export", {})
    
    format_type = args.format or export_config.get("format", "onnx")
    output_path = args.output
    imgsz = args.imgsz or export_config.get("imgsz", 224)
    half = args.half if args.half is not None else export_config.get("half", False)
    simplify = args.simplify if args.simplify is not None else export_config.get("simplify", True)
    opset = args.opset or export_config.get("opset", 12)
    device = args.device or "cpu"
    
    # Initialize exporter
    exporter = ModelExporter(
        model_path=args.model,
        device=device
    )
    
    try:
        if format_type == "onnx":
            results = exporter.export_onnx(
                output_path=output_path,
                imgsz=imgsz,
                half=half,
                simplify=simplify,
                opset=opset
            )
        else:
            results = exporter.export(
                format=format_type,
                output_path=output_path,
                imgsz=imgsz,
                half=half
            )
        
        logger.info("Export completed successfully!")
        logger.info(f"Format: {results['format']}")
        logger.info(f"Output: {results['output_path']}")
        logger.info(f"Size: {results['model_size_mb']} MB")
    
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Classification Suite - Production-grade image classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --data ./data --epochs 100 --batch 16

  # Predict on a single image
  python main.py predict --model runs/classify/run/weights/best.pt --source image.jpg

  # Predict on a folder
  python main.py predict --model best.pt --source ./test_images --save

  # Predict on a video
  python main.py predict --model best.pt --source video.mp4 --output output.mp4

  # Export to ONNX
  python main.py export --model best.pt --format onnx --output model.onnx
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train a YOLOv8 classification model")
    train_parser.add_argument("--model", type=str, help="Model name (e.g., yolov8n-cls.pt)")
    train_parser.add_argument("--data", type=str, help="Path to dataset directory")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch", type=int, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, help="Image size")
    train_parser.add_argument("--workers", type=int, help="Number of worker threads")
    train_parser.add_argument("--device", type=str, help="Device (auto, cpu, cuda, 0, 1, etc.)")
    train_parser.add_argument("--patience", type=int, help="Early stopping patience")
    train_parser.add_argument("--save-period", type=int, help="Save checkpoint every N epochs")
    train_parser.add_argument("--project", type=str, help="Project name for logging")
    train_parser.add_argument("--name", type=str, help="Run name for logging")
    train_parser.add_argument("--save-dir", type=str, help="Directory to save training outputs")
    train_parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    train_parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Don't use pretrained weights")
    train_parser.set_defaults(pretrained=None)
    
    # Predict parser
    predict_parser = subparsers.add_parser("predict", help="Run inference on images/videos")
    predict_parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt)")
    predict_parser.add_argument("--source", type=str, required=True, help="Image, folder, video, or stream source")
    predict_parser.add_argument("--output", type=str, help="Output path for predictions")
    predict_parser.add_argument("--conf", type=float, help="Confidence threshold")
    predict_parser.add_argument("--device", type=str, help="Device (auto, cpu, cuda, 0, 1, etc.)")
    predict_parser.add_argument("--save", action="store_true", help="Save prediction visualizations")
    predict_parser.add_argument("--show", action="store_true", help="Show predictions in window")
    
    # Export parser
    export_parser = subparsers.add_parser("export", help="Export model to different formats")
    export_parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt)")
    export_parser.add_argument("--format", type=str, help="Export format (onnx, torchscript, coreml)")
    export_parser.add_argument("--output", type=str, help="Output path for exported model")
    export_parser.add_argument("--imgsz", type=int, help="Image size for exported model")
    export_parser.add_argument("--half", action="store_true", help="Use FP16 quantization")
    export_parser.add_argument("--no-half", dest="half", action="store_false", help="Don't use FP16")
    export_parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    export_parser.add_argument("--no-simplify", dest="simplify", action="store_false", help="Don't simplify ONNX")
    export_parser.add_argument("--opset", type=int, help="ONNX opset version")
    export_parser.add_argument("--device", type=str, help="Device for export (usually cpu)")
    export_parser.set_defaults(half=None, simplify=None)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute command
    if args.command == "train":
        train_command(args, config)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "export":
        export_command(args, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

