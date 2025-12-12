# YOLOv8 Classification Suite

A production-grade, modular Python software suite for Image Classification using the YOLOv8 framework (Ultralytics).

## Features

- **Professional GUI**: Modern dark-themed PyQt6-based GUI with high-contrast design for optimal legibility and High DPI support
- **Modular Architecture**: Clean separation of concerns with dedicated modules for training, inference, and export
- **Production-Ready**: Proper logging, error handling, type hints, CLI and GUI interfaces
- **Flexible Training**: Configurable training parameters with checkpointing and early stopping
- **Real-Time Monitoring**: Live training progress, loss curves, and validation metrics in GUI with color-coded status indicators
- **Multiple Inference Modes**: Support for single images, image folders, videos (planned), and live streams (planned)
- **Side-by-Side Results**: Image preview panel alongside detailed prediction results with Top-1 and Top-5 classifications
- **Model Export**: Export to ONNX, TorchScript, CoreML, and other formats with configurable options
- **Automatic Device Detection**: Seamless CUDA/CPU handling with manual override options

## Directory Structure

```
SIC/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── trainer.py           # Training module
│   ├── predictor.py         # Inference module
│   └── exporter.py          # Model export module
├── configs/
│   └── config.yaml          # Configuration file
├── main.py                  # Main entry point (CLI)
├── gui_app.py               # Professional GUI application
├── launch_gui.sh            # GUI launcher script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone or navigate to the project directory:
```bash
cd /Users/abinrajbhaskarandevarajan/SIC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## GUI Application

The suite includes a professional-grade GUI application featuring a modern dark theme with high-contrast design for improved legibility and 8K/High DPI support:

### Visual Design:
- **Dark Theme**: Charcoal background (#2b2b2b) with light text (#e0e0e0) for reduced eye strain
- **Green Accents**: Strategic use of green (#4CAF50) for selected elements, progress indicators, and status highlights
- **High Contrast**: Optimized color scheme ensures excellent readability in all lighting conditions
- **Modern UI Elements**: Rounded corners, smooth transitions, and intuitive visual feedback

### Launch GUI:
```bash
python gui_app.py
# or
./launch_gui.sh
```

### GUI Features:

**Training Tab:**
- **Model Configuration**: Selection from yolov8n-cls.pt through yolov8x-cls.pt variants
- **Dataset Browser**: Easy-to-use file browser for selecting dataset directories
- **Training Parameters**: Comprehensive configuration including:
  - Epochs (1-10000)
  - Batch size (1-256)
  - Image size (32-1024, typically 224)
  - Learning rate (0.0001-1.0)
  - Workers, patience, and more
- **Device Selection**: Auto-detection or manual selection (auto, CPU, CUDA, specific GPU)
- **Pretrained Weights**: Toggle for using ImageNet pretrained weights
- **Project Settings**: Customizable project and run names for organization
- **Real-Time Progress**: 
  - Visual progress bar with green accent
  - Live epoch and metrics display (loss, validation accuracy)
  - Color-coded status labels
  - Scrollable training log output
- **Control Buttons**: Start (green) and Stop (red) buttons with hover effects

**Inference Tab:**
- **Model Loading**: Browse and load trained model files (.pt)
- **Input Sources**: 
  - Single image file prediction
  - Batch folder prediction with summary statistics
  - Video file support (placeholder - coming soon)
  - Webcam stream support (placeholder - coming soon)
- **Interactive Preview**: Side-by-side layout with image preview panel
- **Confidence Threshold**: Adjustable slider (0.0-1.0) with real-time value display
- **Save Options**: Checkbox to save prediction visualizations
- **Detailed Results**: 
  - Top-1 prediction with confidence percentage
  - Top-5 predictions with confidence scores
  - For folders: Class distribution summary and detailed per-image results

**Export Tab:**
- **Model Selection**: Browse and select model files for export
- **Export Formats**: ONNX, TorchScript, CoreML
- **Export Configuration**: 
  - Output path specification (auto-suggested based on format)
  - Image size configuration
  - FP16 quantization option (half-precision for smaller models)
  - ONNX simplification option
- **Export Status**: Real-time status messages and completion notifications with file size information

## Dataset Structure for YOLOv8 Classification

YOLOv8 Classification requires a **folder-based structure** where each class has its own subdirectory. The dataset should be organized as follows:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── class3/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class3/
│       ├── image1.jpg
│       └── ...
└── test/                    # Optional
    ├── class1/
    │   └── ...
    ├── class2/
    │   └── ...
    └── class3/
        └── ...
```

### Key Points:

1. **Root Directory**: Contains `train/` and `val/` (and optionally `test/`) folders
2. **Class Folders**: Each class is a subdirectory within `train/` and `val/`
3. **Class Names**: The folder names become the class labels (e.g., "cat", "dog", "bird")
4. **Images**: All images for a class are placed directly in that class's folder
5. **Supported Formats**: JPG, JPEG, PNG, BMP, TIFF, WEBP

### Example:
```
data/
├── train/
│   ├── cat/
│   │   ├── cat_001.jpg
│   │   ├── cat_002.jpg
│   │   └── cat_003.jpg
│   ├── dog/
│   │   ├── dog_001.jpg
│   │   ├── dog_002.jpg
│   │   └── dog_003.jpg
│   └── bird/
│       ├── bird_001.jpg
│       ├── bird_002.jpg
│       └── bird_003.jpg
└── val/
    ├── cat/
    │   ├── cat_004.jpg
    │   └── cat_005.jpg
    ├── dog/
    │   ├── dog_004.jpg
    │   └── dog_005.jpg
    └── bird/
        ├── bird_004.jpg
        └── bird_005.jpg
```

**Important**: The `data` path you provide to the training command should point to the directory containing `train/` and `val/` folders, NOT to the `train/` folder itself.

## Usage

### Training

Train a YOLOv8 classification model:

```bash
python main.py train --data ./data --epochs 100 --batch 16
```

With custom configuration:

```bash
python main.py train \
    --data ./data \
    --model yolov8n-cls.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 224 \
    --device cuda \
    --project my_project \
    --name experiment_1
```

### Inference

**Single Image:**
```bash
python main.py predict \
    --model runs/classify/run/weights/best.pt \
    --source image.jpg \
    --save
```

**Image Folder:**
```bash
python main.py predict \
    --model best.pt \
    --source ./test_images \
    --save \
    --output ./predictions
```

**Video File:**
```bash
python main.py predict \
    --model best.pt \
    --source video.mp4 \
    --output output.mp4 \
    --show
```

**Live Stream (Webcam):**
```bash
python main.py predict \
    --model best.pt \
    --source 0 \
    --show
```

**RTSP Stream:**
```bash
python main.py predict \
    --model best.pt \
    --source rtsp://192.168.1.100:554/stream \
    --show
```

### Export

**Export to ONNX:**
```bash
python main.py export \
    --model best.pt \
    --format onnx \
    --output model.onnx \
    --imgsz 224 \
    --simplify
```

**Export to TorchScript:**
```bash
python main.py export \
    --model best.pt \
    --format torchscript \
    --output model.pt
```

**Export to CoreML:**
```bash
python main.py export \
    --model best.pt \
    --format coreml \
    --output model.mlmodel
```

## Configuration File

The `configs/config.yaml` file allows you to set default parameters. CLI arguments override config file values.

Example configuration:
```yaml
model:
  name: "yolov8n-cls.pt"
  pretrained: true

dataset:
  root: "data"

training:
  epochs: 100
  batch_size: 16
  imgsz: 224
  device: "auto"
```

## Model Variants

YOLOv8 Classification models:
- `yolov8n-cls.pt` - Nano (smallest, fastest)
- `yolov8s-cls.pt` - Small
- `yolov8m-cls.pt` - Medium
- `yolov8l-cls.pt` - Large
- `yolov8x-cls.pt` - Extra Large (most accurate)

## Output Structure

After training, outputs are saved in:
```
runs/classify/
└── run/
    ├── weights/
    │   ├── best.pt          # Best model checkpoint
    │   └── last.pt          # Last epoch checkpoint
    ├── results.png          # Training curves
    ├── confusion_matrix.png # Confusion matrix
    └── ...
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyQt6 (for GUI application)
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full list

## Documentation

- **[GUI Guide](GUI_GUIDE.md)**: Comprehensive guide to using the GUI application
- **[Quick Start Guide](QUICKSTART.md)**: Get started quickly with examples
- **[Directory Structure](DIRECTORY_STRUCTURE.md)**: Project organization details

## License

This project is provided as-is for educational and production use.
