import os
import shutil
import yaml
from sklearn.model_selection import train_test_split

def convert_to_yolov8(
    train_txt_path="./datasets/train.txt",  # Relative path to Train.txt
    images_dir="./datasets/images/train/",  # Relative path to images
    labels_dir="./datasets/labels/train/",  # Relative path to labels
    output_dir="./datasets/output/",  # Relative path for output
    val_split=0.2
):
    """
    Convert existing YOLO dataset to YOLOv8 format
    
    Args:
        train_txt_path: Path to Train.txt file
        images_dir: Directory containing images
        labels_dir: Directory containing label txt files
        output_dir: Where to create YOLOv8 dataset
        val_split: Proportion of data to use for validation
    """
    # Create directory structure
    os.makedirs(os.path.join(output_dir, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val/labels'), exist_ok=True)
    print(f"Train.txt path: {train_txt_path}")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    with open(train_txt_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    print("Image paths from Train.txt:")
    print(image_paths)


    
    # Read image paths from Train.txt
    with open(train_txt_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    # Get base filenames
    image_files = [os.path.basename(path) for path in image_paths]
    
    # Split into train/val
    train_images, val_images = train_test_split(
        image_files,
        test_size=val_split,
        random_state=42
    )
    
    # Copy training files
    for image_file in train_images:
        base_name = os.path.splitext(image_file)[0]
        
        # Copy image
        src_img = os.path.join(images_dir, image_file)
        dst_img = os.path.join(output_dir, 'train/images', image_file)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        
        # Copy label
        label_file = f"{base_name}.txt"
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(output_dir, 'train/labels', label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
    
    # Copy validation files
    for image_file in val_images:
        base_name = os.path.splitext(image_file)[0]
        
        # Copy image
        src_img = os.path.join(images_dir, image_file)
        dst_img = os.path.join(output_dir, 'val/images', image_file)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        
        # Copy label
        label_file = f"{base_name}.txt"
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(output_dir, 'val/labels', label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
    
    # Create data.yaml
    dataset_config = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'license plate'
        }
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(dataset_config, f, sort_keys=False)
    
    print(f"Dataset prepared with {len(train_images)} training and {len(val_images)} validation images")
    print(f"Dataset YAML created at: {os.path.join(output_dir, 'data.yaml')}")

if __name__ == "__main__":
    convert_to_yolov8()
