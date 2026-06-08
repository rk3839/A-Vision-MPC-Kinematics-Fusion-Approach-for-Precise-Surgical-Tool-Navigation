import os
import shutil
import random
from pathlib import Path

def split_yolo_dataset(source_folder, output_folder, train_ratio=0.8, seed=42):
    """
    Split YOLO dataset into train and val sets.
    
    Args:
        source_folder: Path to folder containing images and txt annotation files
        output_folder: Path where train/val split will be created
        train_ratio: Ratio of training data (default 0.8 for 80%)
        seed: Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory structure
    output_path = Path(output_folder)
    dirs = [
        output_path / 'images' / 'train',
        output_path / 'images' / 'val',
        output_path / 'labels' / 'train',
        output_path / 'labels' / 'val'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    source_path = Path(source_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in source_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {source_folder}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle images
    random.shuffle(image_files)
    
    # Calculate split index
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Train set: {len(train_images)} images")
    print(f"Val set: {len(val_images)} images")
    
    # Copy files
    def copy_files(image_list, split_name):
        copied_images = 0
        copied_labels = 0
        missing_labels = []
        
        for img_file in image_list:
            # Copy image
            dst_img = output_path / 'images' / split_name / img_file.name
            shutil.copy2(img_file, dst_img)
            copied_images += 1
            
            # Copy corresponding label file
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                dst_label = output_path / 'labels' / split_name / label_file.name
                shutil.copy2(label_file, dst_label)
                copied_labels += 1
            else:
                missing_labels.append(img_file.name)
        
        print(f"\n{split_name.upper()} set:")
        print(f"  Copied {copied_images} images")
        print(f"  Copied {copied_labels} labels")
        if missing_labels:
            print(f"  Warning: {len(missing_labels)} images without labels")
            if len(missing_labels) <= 5:
                print(f"  Missing labels for: {', '.join(missing_labels)}")
    
    # Process train and val sets
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    
    print(f"\n✓ Dataset split complete!")
    print(f"Output directory: {output_folder}")
    print("\nFolder structure:")
    print(f"{output_folder}/")
    print("├── images/")
    print("│   ├── train/")
    print("│   └── val/")
    print("└── labels/")
    print("    ├── train/")
    print("    └── val/")


if __name__ == "__main__":
    # Configuration
    SOURCE_FOLDER = r"C:\Users\vergi\Downloads\corrected_annonation_exp_dataset\obj_train_data"  # Change this to your dataset folder
    OUTPUT_FOLDER = r"C:\Users\vergi\Downloads\corrected_annonation_exp_dataset\output"         # Change this to desired output folder
    TRAIN_RATIO = 0.8                        # 80% train, 20% val
    
    # Run the split
    split_yolo_dataset(
        source_folder=SOURCE_FOLDER,
        output_folder=OUTPUT_FOLDER,
        train_ratio=TRAIN_RATIO,
        seed=42  # For reproducibility
    )
