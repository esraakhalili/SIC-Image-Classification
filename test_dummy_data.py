#!/usr/bin/env python3
"""
Quick test script to verify the dummy dataset structure.
"""

from pathlib import Path

def verify_dataset(data_path: str = "data"):
    """Verify that the dataset structure is correct."""
    data_path = Path(data_path)
    
    print(f"Verifying dataset structure: {data_path.absolute()}\n")
    
    # Check train directory
    train_path = data_path / "train"
    if not train_path.exists():
        print("❌ ERROR: train/ directory not found!")
        return False
    
    # Check val directory
    val_path = data_path / "val"
    if not val_path.exists():
        print("❌ ERROR: val/ directory not found!")
        return False
    
    # Get classes from train directory
    train_classes = [d.name for d in train_path.iterdir() if d.is_dir()]
    val_classes = [d.name for d in val_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(train_classes)} classes in train/: {train_classes}")
    print(f"Found {len(val_classes)} classes in val/: {val_classes}\n")
    
    if set(train_classes) != set(val_classes):
        print("⚠️  WARNING: Classes in train/ and val/ don't match!")
    
    # Count images in each class
    print("Image counts:")
    total_train = 0
    total_val = 0
    
    for class_name in sorted(train_classes):
        class_train_path = train_path / class_name
        class_val_path = val_path / class_name
        
        train_images = list(class_train_path.glob("*.*"))
        val_images = list(class_val_path.glob("*.*")) if class_val_path.exists() else []
        
        train_count = len(train_images)
        val_count = len(val_images)
        
        total_train += train_count
        total_val += val_count
        
        print(f"  {class_name}:")
        print(f"    train/: {train_count} images")
        print(f"    val/:   {val_count} images")
    
    print(f"\nTotal: {total_train} training images, {total_val} validation images")
    print(f"Grand total: {total_train + total_val} images\n")
    
    if total_train == 0:
        print("❌ ERROR: No training images found!")
        return False
    
    if total_val == 0:
        print("⚠️  WARNING: No validation images found!")
    
    print("✅ Dataset structure looks good!")
    print(f"\nYou can now test training with:")
    print(f"  python main.py train --data {data_path} --epochs 10 --batch 8")
    
    return True

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data"
    verify_dataset(data_path)

