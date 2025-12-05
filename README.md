# YOLOv8 Classification Suite

A production-grade, modular Python software suite for Image Classification using the YOLOv8 framework (Ultralytics).

## Features

- **Professional GUI**: Modern, interactive PyQt6-based GUI with 8K/High DPI support
- **Modular Architecture**: Clean separation of concerns with dedicated modules for training, inference, and export
- **Production-Ready**: Proper logging, error handling, type hints, CLI and GUI interfaces
- **Flexible Training**: Configurable training parameters with checkpointing and early stopping
- **Real-Time Monitoring**: Live training progress, loss curves, and validation metrics in GUI
- **Multiple Inference Modes**: Support for single images, image folders, videos, and live streams
- **Interactive Preview**: Image preview with prediction overlays and detailed results
- **Model Export**: Export to ONNX, TorchScript, CoreML, and other formats
- **Automatic Device Detection**: Seamless CUDA/CPU handling

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


## Dataset Instructions

The full dataset is not included in this repository due to size. 
To run the project:

1. Download the dataset from this link: 
   [Download data.zip](https://1drv.ms/u/c/f62b56aaaf734aa0/EXEiF17_VpRDvmyzzYUqbloByH5GdvHa4jpRq2WOOUPXgg?e=4UtUKN)

2. Extract the zip and place the `data/` folder in the root of the SIC project:
   SIC/data/

3. Ensure the folder structure is:
   SIC/data/train/cat/
   SIC/data/train/dog/
   SIC/data/train/bird/
   SIC/data/val/cat/
   SIC/data/val/dog/
   SIC/data/val/bird/


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

The suite includes a professional-grade GUI application with 8K/High DPI support:

### Launch GUI:
```bash
python gui_app.py
# or
./launch_gui.sh
```

### GUI Features:

**Training Tab:**
- Model selection (yolov8n-cls.pt through yolov8x-cls.pt)
- Dataset path browser
- Training parameter configuration (epochs, batch size, learning rate, etc.)
- Real-time training progress with live updates
- Training logs and status messages
- Device selection (auto, CPU, CUDA)

**Inference Tab:**
- Model loading and selection
- Multiple input sources (Image, Folder, Video, Webcam)
- Interactive image preview
- Confidence threshold slider
- Detailed prediction results (Top-1 and Top-5)
- Batch folder prediction with summary statistics

**Export Tab:**
- Model export to ONNX, TorchScript, CoreML
- Export configuration (image size, FP16 quantization, simplification)
- Export status and progress tracking

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
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full list

## License

This project is provided as-is for educational and production use.

