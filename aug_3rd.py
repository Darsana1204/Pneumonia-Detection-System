import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm # For progress bar
import random
import uuid # Unique filenames

# Set a fixed random seed for consistent results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define input and output paths
input_folder = Path("D:/Miniproject/Database/train/NORMAL") # Path to original images
output_folder = Path("D:/Miniproject/Database/train/NORMAL_augmented") # Augmented images saved here
output_folder.mkdir(parents=True, exist_ok=True) # Create directory if not exists

# Get list of original images
original_images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png")) + list(input_folder.glob("*.jpeg"))
num_original = len(original_images)
target_count = 3875 # Target: Match Pneumonia class
total_to_generate = target_count - num_original

print(f"\n Starting augmentation...")
print(f" Original Normal Images: {num_original}")
print(f" Target Count: {target_count}")
print(f" Saving augmented images to: {output_folder}\n")

# Define augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5), # Left-right flip (valid for lungs)
    A.RandomBrightnessContrast(p=0.3), # Adjust brightness & contrast
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3) # Small shifts, scaling, rotations
], p=1.0) # Apply all selected augmentations

# Print augmentation methods
print(" Applied Augmentations:")
print(" Horizontal Flip (50%)")
print(" Random Brightness & Contrast (30%)")
print(" Shift, Scale, Rotate (30%)\n")

# Start augmentation process
augmented_count = 0
with tqdm(total=total_to_generate, desc=" Augmenting Images", unit="img") as pbar:
    while augmented_count < total_to_generate:
        for img_path in original_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue # Skip if image is not loaded

            # Convert BGR to RGB (Albumentations expects RGB format)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply augmentation
            augmented_data = augmentations(image=image)
            augmented = augmented_data["image"]

            # Save augmented image with unique filename
            aug_filename = f"{img_path.stem}_aug_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(str(output_folder / aug_filename), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

            augmented_count += 1
            pbar.update(1) # Update progress bar

            # Stop when enough images are generated
            if augmented_count >= total_to_generate:
                break

# Final message
print(f"\n Augmentation complete!")
print(f" Final Normal Images Count: {num_original + augmented_count}")
print(f" Augmented Images Saved In: {output_folder}")
print(f" Total Augmented Images Created: {augmented_count} (Target: {total_to_generate})")
