# GUI Application Guide

## Overview

The YOLOv8 Classification Suite includes a professional-grade GUI application built with PyQt6, featuring:

- **8K/High DPI Support**: Fully scalable interface for high-resolution displays
- **Modern Design**: Clean, professional interface with intuitive navigation
- **Real-Time Updates**: Live training progress and inference results
- **Interactive Preview**: Image preview with prediction overlays

## Launching the GUI

### Method 1: Direct Python execution
```bash
python gui_app.py
```

### Method 2: Using launcher script
```bash
./launch_gui.sh
```

### Method 3: From Python
```python
from gui_app import main
main()
```

## Interface Overview

The GUI consists of three main tabs:

### 1. Training Tab

**Model Configuration:**
- **Model**: Select from yolov8n-cls.pt (nano) to yolov8x-cls.pt (extra large)
- **Dataset Path**: Browse and select your dataset directory
- **Device**: Choose auto, CPU, or specific CUDA device
- **Pretrained**: Toggle to use pretrained ImageNet weights

**Training Parameters:**
- **Epochs**: Number of training epochs (1-10000)
- **Batch Size**: Batch size for training (1-256)
- **Image Size**: Input image size (32-1024, typically 224)
- **Workers**: Number of data loading workers (0-32)
- **Learning Rate**: Initial learning rate (0.0001-1.0)
- **Patience**: Early stopping patience

**Project Settings:**
- **Project Name**: Name for organizing training runs
- **Run Name**: Specific name for this training run

**Training Progress:**
- Real-time progress bar
- Current epoch and metrics display
- Training log output
- Start/Stop controls

**Usage:**
1. Select model variant
2. Browse and select dataset directory
3. Configure training parameters
4. Click "Start Training"
5. Monitor progress in real-time
6. Training outputs saved to `runs/classify/[project]/[run]/weights/`

### 2. Inference Tab

**Model Selection:**
- Browse and select trained model file (.pt)
- Supports both pretrained and custom trained models

**Input Source:**
- **Image**: Single image file prediction
- **Folder**: Batch prediction on folder of images
- **Video**: Video file prediction (coming soon)
- **Webcam**: Live webcam stream (coming soon)

**Settings:**
- **Confidence Threshold**: Adjustable slider (0.0-1.0)
- **Save Predictions**: Option to save prediction visualizations

**Results Display:**
- **Preview Panel**: Shows input image with prediction overlay
- **Results Panel**: Detailed prediction results including:
  - Top-1 prediction with confidence
  - Top-5 predictions with confidence scores
  - For folders: Summary statistics and class distribution

**Usage:**
1. Load trained model
2. Select source type (Image/Folder/Video/Webcam)
3. Browse and select source
4. Adjust confidence threshold if needed
5. Click "Run Prediction"
6. View results in preview and results panels

### 3. Export Tab

**Model Selection:**
- Browse and select model file to export

**Export Settings:**
- **Format**: Choose ONNX, TorchScript, or CoreML
- **Output Path**: Specify save location
- **Image Size**: Input size for exported model
- **FP16 Quantization**: Enable half-precision (smaller file size)
- **Simplify ONNX**: Simplify ONNX model graph (ONNX only)

**Status:**
- Export progress messages
- Completion status with file size

**Usage:**
1. Select model to export
2. Choose export format
3. Specify output path (auto-suggested based on format)
4. Configure export options
5. Click "Export Model"
6. Wait for completion and verify output file

## Keyboard Shortcuts

- **Ctrl+Q**: Quit application
- **Ctrl+O**: Open file dialogs (context-dependent)
- **Tab**: Navigate between fields
- **Enter**: Activate primary button in current tab

## High DPI / 8K Display Support

The GUI automatically detects and scales for high DPI displays:

- **Automatic Scaling**: Enabled by default
- **High DPI Pixmaps**: Crisp icons and images at any resolution
- **Font Scaling**: Text scales appropriately
- **Layout Adaptation**: Interface adapts to screen size

For manual control, set environment variable:
```bash
export QT_SCALE_FACTOR=1.5  # Adjust scale factor
python gui_app.py
```

## Tips and Best Practices

### Training:
- Start with smaller models (yolov8n-cls.pt) for faster iteration
- Use pretrained weights for better initial performance
- Monitor training logs for early stopping opportunities
- Adjust batch size based on available GPU memory

### Inference:
- Lower confidence threshold for more permissive predictions
- Use folder prediction for batch processing
- Save predictions to review model performance visually

### Export:
- ONNX is most widely supported format
- Enable FP16 quantization for smaller models (may have slight accuracy loss)
- Simplify ONNX for better compatibility with inference engines

## Troubleshooting

### GUI won't launch:
- Ensure PyQt6 is installed: `pip install PyQt6`
- Check Python version (3.8+ required)
- Verify all dependencies: `pip install -r requirements.txt`

### High DPI display issues:
- GUI should auto-detect, but if text/icons are too small:
  - Set `QT_SCALE_FACTOR` environment variable
  - Or adjust system display scaling

### Training doesn't start:
- Verify dataset path is correct
- Check dataset structure matches requirements
- Ensure sufficient disk space for checkpoints

### Inference errors:
- Verify model file exists and is valid
- Check image format is supported (JPG, PNG, BMP, PPM)
- Ensure model matches task (classification, not detection)

## Advanced Features

### Custom Themes:
Modify `apply_styles()` method in `MainWindow` class to customize colors and styling.

### Batch Operations:
Use folder inference for processing multiple images efficiently.

### Model Comparison:
Train multiple models with different settings and compare results in the inference tab.

## Performance Notes

- **GPU Acceleration**: Automatically detected and used when available
- **Memory Management**: Large batch sizes may require GPU with more VRAM
- **Multi-threading**: Training and inference run in separate threads to keep UI responsive

## Future Enhancements

Planned features:
- Real-time training curve visualization
- Video and webcam stream support in inference tab
- Model comparison and benchmarking tools
- Dataset visualization and statistics
- Advanced augmentation preview

