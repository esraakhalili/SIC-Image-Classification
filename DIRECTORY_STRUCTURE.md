# Directory Structure

```
SIC/
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
├── DIRECTORY_STRUCTURE.md     # This file
├── requirements.txt           # Python dependencies
├── main.py                    # Main entry point (CLI)
├── configs/
│   └── config.yaml            # Configuration file
└── src/
    ├── __init__.py            # Package initialization
    ├── trainer.py             # Training module
    ├── predictor.py           # Inference module
    └── exporter.py            # Model export module
```

## File Descriptions

### Root Level

- **main.py**: Command-line interface entry point. Handles training, inference, and export commands.
- **requirements.txt**: Python package dependencies for the project.
- **README.md**: Comprehensive documentation including usage examples and dataset structure.
- **.gitignore**: Git ignore patterns for Python projects and YOLOv8 outputs.

### configs/

- **config.yaml**: YAML configuration file with default settings for model, dataset, training, logging, and export parameters.

### src/

- **__init__.py**: Package initialization file with version and author information.
- **trainer.py**: `ClassificationTrainer` class for training YOLOv8 classification models with callbacks and checkpointing.
- **predictor.py**: `ClassificationPredictor` class for inference on images, folders, videos, and streams.
- **exporter.py**: `ModelExporter` class for exporting models to ONNX, TorchScript, CoreML, and other formats.

## Generated Directories (after running)

After training or inference, the following directories will be created:

```
SIC/
├── runs/
│   └── classify/
│       └── run/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.png
│           └── confusion_matrix.png
└── predictions/              # (if saving predictions)
    └── pred_*.jpg
```
