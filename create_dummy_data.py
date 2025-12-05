#!/usr/bin/env python3
"""
Script to create dummy dataset for testing YOLOv8 Classification.
Creates a simple dataset with colored images for different classes.
"""

import os
import struct
from pathlib import Path
import random

def create_simple_image(size: tuple = (224, 224), color: tuple = None, label: str = "") -> bytes:
    """
    Create a simple PPM image (P6 format) with a solid color.
    PPM is a simple, uncompressed image format that doesn't require external libraries.
    
    Args:
        size: Image dimensions (width, height)
        color: RGB color tuple (if None, random color)
        label: Label name (for filename reference)
        
    Returns:
        Bytes representing the PPM image
    """
    width, height = size
    if color is None:
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    # PPM P6 format header
    header = f"P6\n{width} {height}\n255\n".encode('ascii')
    
    # Create image data (all pixels same color)
    pixel = struct.pack('BBB', color[0], color[1], color[2])
    image_data = pixel * (width * height)
    
    return header + image_data

def create_dummy_dataset(
    root_dir: str = "data",
    classes: list = None,
    train_images_per_class: int = 50,
    val_images_per_class: int = 10,
    image_size: tuple = (224, 224)
):
    """
    Create a dummy dataset with the required folder structure.
    
    Args:
        root_dir: Root directory for the dataset
        classes: List of class names (if None, uses default classes)
        train_images_per_class: Number of training images per class
        val_images_per_class: Number of validation images per class
        image_size: Size of generated images
    """
    if classes is None:
        classes = ["cat", "dog", "bird"]
    
    root_path = Path(root_dir)
    train_path = root_path / "train"
    val_path = root_path / "val"
    
    # Define colors for each class (for visual distinction)
    class_colors = {
        "cat": (255, 150, 150),    # Light red
        "dog": (150, 255, 150),    # Light green
        "bird": (150, 150, 255),   # Light blue
    }
    
    print(f"Creating dummy dataset in: {root_path.absolute()}")
    print(f"Classes: {classes}")
    print(f"Training images per class: {train_images_per_class}")
    print(f"Validation images per class: {val_images_per_class}")
    print("-" * 60)
    
    # Create training set
    for class_name in classes:
        class_train_path = train_path / class_name
        class_train_path.mkdir(parents=True, exist_ok=True)
        
        color = class_colors.get(class_name, None)
        
        print(f"Creating {train_images_per_class} training images for '{class_name}'...", end=" ")
        for i in range(train_images_per_class):
            # Add slight variation to color
            if color:
                varied_color = tuple(
                    max(0, min(255, c + random.randint(-30, 30)))
                    for c in color
                )
            else:
                varied_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            
            # Create image as PPM, then convert to JPEG using a simple method
            # For simplicity, we'll save as PPM first, but YOLOv8 can read PPM
            # Actually, let's try to create a simple JPEG-like structure or use PPM
            # Since YOLOv8 supports various formats, let's use PPM which is simple
            
            img_data = create_simple_image(
                size=image_size,
                color=varied_color,
                label=class_name
            )
            
            # Save as PPM (YOLOv8 supports this)
            img_path = class_train_path / f"{class_name}_{i+1:03d}.ppm"
            with open(img_path, 'wb') as f:
                f.write(img_data)
        
        print(f"✓ ({train_images_per_class} images)")
    
    # Create validation set
    for class_name in classes:
        class_val_path = val_path / class_name
        class_val_path.mkdir(parents=True, exist_ok=True)
        
        color = class_colors.get(class_name, None)
        
        print(f"Creating {val_images_per_class} validation images for '{class_name}'...", end=" ")
        for i in range(val_images_per_class):
            # Add slight variation to color
            if color:
                varied_color = tuple(
                    max(0, min(255, c + random.randint(-30, 30)))
                    for c in color
                )
            else:
                varied_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            
            img_data = create_simple_image(
                size=image_size,
                color=varied_color,
                label=class_name
            )
            
            img_path = class_val_path / f"{class_name}_{i+1:03d}.ppm"
            with open(img_path, 'wb') as f:
                f.write(img_data)
        
        print(f"✓ ({val_images_per_class} images)")
    
    print("-" * 60)
    print("Dataset creation completed!")
    print(f"\nDataset structure:")
    print(f"{root_dir}/")
    print(f"├── train/")
    for class_name in classes:
        print(f"│   ├── {class_name}/ ({train_images_per_class} images)")
    print(f"└── val/")
    for class_name in classes:
        print(f"    ├── {class_name}/ ({val_images_per_class} images)")
    
    total_train = len(classes) * train_images_per_class
    total_val = len(classes) * val_images_per_class
    print(f"\nTotal images: {total_train + total_val} ({total_train} train, {total_val} val)")
    print(f"\nNote: Images are saved as .ppm format (YOLOv8 supports this).")
    print(f"You can test training with: python main.py train --data {root_dir} --epochs 10 --batch 8")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create dummy dataset for YOLOv8 Classification")
    parser.add_argument("--root", type=str, default="data", help="Root directory for dataset")
    parser.add_argument("--classes", type=str, nargs="+", default=["cat", "dog", "bird"],
                        help="Class names")
    parser.add_argument("--train", type=int, default=50, help="Training images per class")
    parser.add_argument("--val", type=int, default=10, help="Validation images per class")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224],
                        help="Image size (width height)")
    
    args = parser.parse_args()
    
    create_dummy_dataset(
        root_dir=args.root,
        classes=args.classes,
        train_images_per_class=args.train,
        val_images_per_class=args.val,
        image_size=tuple(args.size)
    )
