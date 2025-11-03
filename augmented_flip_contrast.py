import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm  # Progress bar
from collections import defaultdict  # Track augmentation usage
import sys

# ✅ Fix Unicode Encoding Issues for Windows
sys.stdout.reconfigure(encoding='utf-8')

# 🔹 Define Paths
input_folder = Path("D:/Miniproject/Database/train/NORMAL")  # Original Normal images
output_folder = Path("D:/Miniproject/Database/train/NORMAL_augmented")  # Augmented images
output_folder.mkdir(parents=True, exist_ok=True)  # Create output folder if not exists

# 🔹 Get List of Original Images
original_images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png")) + list(input_folder.glob("*.jpeg"))
num_original = len(original_images)
target_count = num_original + 2519  # Total images after augmentation
total_to_generate = target_count - num_original  # Required augmented images (2519)

print(f"\n🚀 Starting augmentation...")
print(f"🔹 Original Normal Images: {num_original}")
print(f"🔹 Total Augmented Images Required: {total_to_generate}")
print(f"📂 Saving augmented images to: {output_folder}\n")

# 🔹 Define Augmentation Pipeline (Vertical Flip + Contrast)
augmentations = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

# ✅ Print Applied Augmentations
print("🔄 Augmentation Methods Applied:")
print(" - Vertical Flip (50%)")
print(" - Random Brightness/Contrast (30%)\n")

# 🔹 Track Unique Augmented Images
augmentation_tracker = defaultdict(int)
augmented_count = 0

print("\n⚡ Augmenting images...")

# ✅ Use Progress Bar for Augmentation
with tqdm(total=total_to_generate, desc="📸 Progress", unit="img") as pbar:
    while augmented_count < total_to_generate:
        for img_path in original_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue  # Skip if image can't be read

            # Apply Augmentation
            augmented_data = augmentations(image=image)
            augmented = augmented_data["image"]

            # Save Augmented Image
            aug_filename = f"{img_path.stem}_aug{augmented_count}.jpg"
            cv2.imwrite(str(output_folder / aug_filename), augmented)

            augmented_count += 1
            pbar.update(1)  # Update progress bar

            # Track Augmentation Methods
            for transform in augmentations.transforms:
                augmentation_tracker[type(transform).__name__] += 1

            # Stop when we reach the required number
            if augmented_count >= total_to_generate:
                break

# ✅ Final Message
print(f"\n✅ Augmentation Complete!")
print(f"📊 Final Normal Images Count: {num_original + augmented_count}")
print(f"📂 Augmented Images Saved In: {output_folder}")

# 🔹 Print Augmentation Statistics
print("\n📊 Final Augmentation Counts:")
for method, count in augmentation_tracker.items():
    print(f" - {method}: {count} times")

print(f"\n🎯 Total Augmented Images Created: {augmented_count} (Should match {total_to_generate})")

