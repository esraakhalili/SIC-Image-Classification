# Quick Start Guide

Follow these steps to get started with the YOLOv8 Classification Suite.

## Step 1: Install Dependencies

First, install all required Python packages:

```bash
cd /Users/abinrajbhaskarandevarajan/SIC
pip install -r requirements.txt
```

**Note**: If you have CUDA available and want GPU acceleration, make sure PyTorch with CUDA support is installed. The requirements.txt will install the CPU version by default. For CUDA, visit [pytorch.org](https://pytorch.org/) to get the appropriate installation command.

## Step 2: Verify Installation

Test that everything is set up correctly:

```bash
python main.py --help
```

You should see the help menu with three commands: `train`, `predict`, and `export`.

## Step 3: Prepare Your Dataset

Organize your images in the following structure:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── class3/
│       └── ...
└── val/
    ├── class1/
    │   └── ...
    ├── class2/
    │   └── ...
    └── class3/
        └── ...
```

**Important**: 
- Each class should be a folder inside `train/` and `val/`
- The folder names become your class labels
- Place images directly in the class folders (not in subfolders)

## Step 4: Train Your Model

Start training with default settings:

```bash
python main.py train --data ./data --epochs 50 --batch 16
```

Or customize training parameters:

```bash
python main.py train \
    --data ./data \
    --model yolov8n-cls.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 224 \
    --device cuda \
    --project my_classification_project \
    --name experiment_1
```

**Training outputs** will be saved in `runs/classify/run/weights/`:
- `best.pt` - Best model checkpoint
- `last.pt` - Last epoch checkpoint

## Step 5: Test Inference

After training, test your model on a single image:

```bash
python main.py predict \
    --model runs/classify/run/weights/best.pt \
    --source path/to/test_image.jpg \
    --save
```

Test on a folder of images:

```bash
python main.py predict \
    --model runs/classify/run/weights/best.pt \
    --source ./test_images \
    --save \
    --output ./predictions
```

## Step 6: Export Model (Optional)

Export your trained model to ONNX for deployment:

```bash
python main.py export \
    --model runs/classify/run/weights/best.pt \
    --format onnx \
    --output model.onnx \
    --simplify
```

## Common Use Cases

### Quick Test with Pretrained Model

If you just want to test inference without training:

```bash
python main.py predict \
    --model yolov8n-cls.pt \
    --source image.jpg
```

This uses the pretrained ImageNet model.

### Video Inference

```bash
python main.py predict \
    --model best.pt \
    --source video.mp4 \
    --output output.mp4 \
    --show
```

### Webcam Stream

```bash
python main.py predict \
    --model best.pt \
    --source 0 \
    --show
```

Press 'q' to quit the stream.

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python main.py train --data ./data --batch 8
```

### Dataset Not Found
Make sure your dataset path is correct and follows the folder structure exactly.

### Model File Not Found
Check that training completed successfully and the model exists at the specified path.

## Next Steps

- Experiment with different model sizes (yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt)
- Tune hyperparameters in `configs/config.yaml`
- Try different image sizes (imgsz parameter)
- Use data augmentation settings in the config file

