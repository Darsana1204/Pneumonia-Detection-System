import cv2
import os

def enhance_image_with_clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Enhances an image using CLAHE and returns the enhanced image."""
    print(f"Enhancing image: {image_path}")  # Debug message
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
   
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return None

    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l_channel = clahe.apply(l_channel)

    # Merge enhanced L-channel back with A and B channels
    enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    return enhanced_image

def process_folder(input_folder, output_folder):
    """Processes all images in the input folder and subfolders using CLAHE and saves them in the output folder."""
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    if not os.path.exists(input_folder):
        print("Error: Input folder does not exist.")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for root, _, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)
        print(f"Processing directory: {root}")

        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, file)

                # Enhance image and save the output
                enhanced_image = enhance_image_with_clahe(input_path)
                if enhanced_image is not None:
                    cv2.imwrite(output_path, enhanced_image)
                    print(f"Saved enhanced image: {output_path}")
                else:
                    print(f"Skipping invalid image: {input_path}")
            else:
                print(f"Skipping non-image file: {file}")

# Define input and output folders
input_folder = os.path.expanduser("D:/Miniproject/Database")  # Path to your input folder (Database on Desktop)
output_folder = os.path.expanduser("D:/Miniproject/enhanced_Database")  # Path to save enhanced images

# Run the processing function
process_folder(input_folder, output_folder)
