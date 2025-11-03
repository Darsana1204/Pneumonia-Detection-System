import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm  # Progress bar
from collections import defaultdict  # Track augmentation usage

# Define paths
input_folder = Path("D:/Miniproject/Database/train/NORMAL")  
output_folder = Path("D:/Miniproject/Database/train")  # Augmented images added in NORMAL
output_folder.mkdir(parents=True, exist_ok=True)

# Get list of original images
original_images = list(input_folder.glob(".jpg")) + list(input_folder.glob(".png")) + list(input_folder.glob("*.jpeg"))
num_original = len(original_images)
target_count = 3875  # Match Pneumonia class
total_to_generate = target_count - num_original

print(f"\n Starting augmentation...")
print(f" Original Normal Images: {num_original}")
print(f" Target Count: {target_count}")
print(f" Saving augmented images to: {output_folder}\n")

# Define augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.3),
    A.RandomGamma(p=0.3),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.ISONoise(p=0.5)
    ], p=0.5),
])

# Print augmentation methods
print(" Augmentation Methods Applied:")
print(" Horizontal Flip (50%)")
print(" Rotation (-30° to 30°) (50%)")
print(" Random Brightness/Contrast (30%)")
print(" Gaussian Blur (30%)")
print(" Random Gamma Correction (30%)")
print(" Noise Injection (Gaussian/ISO) (50%)\n")

# Track unique augmented images
augmentation_tracker = defaultdict(int)
augmented_count = 0

print("\n Augmenting images...")

# Use tqdm progress bar
with tqdm(total=total_to_generate, desc=" Progress", unit="img") as pbar:
    while augmented_count < total_to_generate:
        for img_path in original_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Apply augmentation
            augmented_data = augmentations(image=image)
            augmented = augmented_data["image"]

            # Save augmented image
            aug_filename = f"{img_path.stem}_aug{augmented_count}.jpg"
            cv2.imwrite(str(output_folder / aug_filename), augmented)

            augmented_count += 1
            pbar.update(1)  # Update progress bar

            # Track unique augmentation use
            for transform in augmentations.transforms:
                if isinstance(transform, A.OneOf):
                    chosen = transform.transforms[np.random.randint(len(transform.transforms))]
                else:
                    chosen = transform
                augmentation_tracker[chosen.__class__.__name__] += 1


            # Show live augmented image count
            print(f" Unique Augmented Images: {augmented_count}/{total_to_generate}", end="\r")

            if augmented_count >= total_to_generate:
                break

# Final message
print(f"\n Augmentation complete!")
print(f" Final Normal Images Count: {num_original + augmented_count}")
print(f" Augmented Images Saved In: {output_folder}")

# Print augmentation statistics
print("\n Final Augmentation Counts (Each image had multiple transforms applied):")
for method, count in augmentation_tracker.items():
    print(f"🔹 {method}: {count} times")

print(f"\n Total Augmented Images Created: {augmented_count} (Should match {total_to_generate})")