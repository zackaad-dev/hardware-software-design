import os
import cv2
import random
import argparse
import numpy as np
import albumentations as A
from pathlib import Path

def augment_dataset(input_dir, output_dir, target_total, target_resolution=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
    
    if not image_files:
        raise ValueError("No images found in the input directory.")

    processed_images = []

    if target_resolution:
        width, height = target_resolution
        resize_transform = A.Resize(height=height, width=width)
    else:
        resize_transform = None

    augmentation_pipeline = A.Compose([
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.2, 0.2),
            rotate=(-45, 45),
            p=0.8
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.3, 
            contrast=0.3, 
            saturation=0.3, 
            hue=0.2, 
            p=0.8
        ),
        A.GaussNoise(std_range=(0.05, 0.1), p=0.4)
    ])

    current_count = 0

    for img_file in image_files:
        if current_count >= target_total:
            break
            
        image = cv2.imread(str(img_file))
        if image is None:
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if resize_transform:
            processed = resize_transform(image=image)['image']
        else:
            processed = image
            
        processed_images.append(processed)
        
        out_name = output_path / f"orig_{current_count}_{img_file.name}"
        cv2.imwrite(str(out_name), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        current_count += 1

    if not processed_images:
        raise ValueError("Failed to load any valid images.")

    while current_count < target_total:
        base_image = random.choice(processed_images)
        
        augmented = augmentation_pipeline(image=base_image)['image']
        
        out_name = output_path / f"aug_{current_count}.jpg"
        cv2.imwrite(str(out_name), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        current_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image dataset augmentation script.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input folder")
    parser.add_argument("-t", "--total", type=int, required=True, help="Target total number of images")
    parser.add_argument("--width", type=int, default=240, help="Target width (default: 240)")
    parser.add_argument("--height", type=int, default=240, help="Target height (default: 240)")
    parser.add_argument("--no-resize", action="store_true", help="Disable image resizing")

    args = parser.parse_args()

    input_path_resolved = Path(args.input).resolve()
    output_dir = f"{input_path_resolved}-augmented"

    target_resolution = None if args.no_resize else (args.width, args.height)

    augment_dataset(args.input, output_dir, args.total, target_resolution)
