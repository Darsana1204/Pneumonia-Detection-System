import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from docx import Document

# Define paths
base_folder = r"D:\Miniproject\InceptionV3"
os.makedirs(base_folder, exist_ok=True)

feature_npz_path = os.path.join(base_folder, "FEATURES_IV3.npz")
feature_excel_path = os.path.join(base_folder, "FEATURES_IV3.xlsx")
analysis_doc_path = os.path.join(base_folder, "ANALYSIS_IV3.docx")
heatmap_path = os.path.join(base_folder, "HEATMAP_IV3.png")
architecture_path = os.path.join(base_folder, "ARCHITECTURE_IV3.png")

data_path = r"D:\Miniproject\enhanced_Database\train"
classes = ["NORMAL", "PNEUMONIA"]

# Load pre-trained InceptionV3 model (without top layers)
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features
def extract_features(directory, label):
    features = []
    image_names = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        try:
            img = load_img(img_path, target_size=(299, 299))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
            feature = base_model.predict(img_array)
            if feature is not None:
                features.append(feature.flatten())
                image_names.append(filename)
                labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return np.array(features), image_names, labels

# Extract features for both classes
features_normal, names_normal, labels_normal = extract_features(os.path.join(data_path, "NORMAL"), "NORMAL")
features_pneumonia, names_pneumonia, labels_pneumonia = extract_features(os.path.join(data_path, "PNEUMONIA"), "PNEUMONIA")

# Combine all extracted features
all_features = np.vstack((features_normal, features_pneumonia))
all_image_names = names_normal + names_pneumonia
all_labels = labels_normal + labels_pneumonia

# Save features to NPZ
np.savez(feature_npz_path, features=all_features, labels=all_labels)
print(f"Features saved to {feature_npz_path}")

# Save features to Excel
df_features = pd.DataFrame(all_features, index=all_image_names)
df_features.columns = [f"Feature_{i+1}" for i in range(all_features.shape[1])]
df_features.insert(0, "Label", all_labels)
df_features.to_excel(feature_excel_path, index=False)
print(f"Features saved to {feature_excel_path}")

# Apply PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

# Sort data by labels to group NORMAL and PNEUMONIA samples
sorted_indices = np.argsort(all_labels)
sorted_features = reduced_features[sorted_indices]
sorted_labels = np.array(all_labels)[sorted_indices]

# Generate heatmap with NORMAL and PNEUMONIA on x-axis
plt.figure(figsize=(8, 6))
ax = sns.heatmap(sorted_features, cmap="GnBu", cbar=True, xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=False)

# Save heatmap
plt.xlabel("Categories")
plt.ylabel("Samples")
plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Heatmap saved at {heatmap_path}")

# Save model specifications in Word file
doc = Document()
doc.add_heading("InceptionV3 Model Analysis", level=1)

# Count total layers and layer types
layer_types = {}
for layer in base_model.layers:
    layer_type = layer.__class__.__name__
    layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

doc.add_paragraph(f"Total Number of Layers: {len(base_model.layers)}")
doc.add_heading("Layer Type Counts", level=2)
for layer_type, count in layer_types.items():
    doc.add_paragraph(f"{layer_type}: {count} layers")

doc.save(analysis_doc_path)
print(f"Model analysis saved at {analysis_doc_path}")

# Create architecture diagram
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# Define layers with specifications
layers = [
    ("Input", "(299,299,3)", (3, 12)),
    ("Conv2D", "64 filters, 3x3", (3, 10)),
    ("MaxPooling2D", "2x2 pool", (3, 8)),
    ("Inception Module", "Multiple convolutions", (3, 6)),
    ("GlobalAvgPooling2D", "Feature Compression", (3, 4)),
    ("Output", "2048 features", (3, 2))
]

# Draw blocks for each layer
for layer_name, spec, (x, y) in layers:
    rect = patches.FancyBboxPatch((x, y), 4, 1.5, boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightblue")
    ax.add_patch(rect)
    ax.text(x + 2, y + 1, f"{layer_name}\n{spec}", fontsize=10, ha='center', va='center', fontweight='bold')

# Draw arrows between blocks
for i in range(len(layers) - 1):
    x1, y1 = layers[i][2]
    x_arrow = x1 + 2
    y_start = y1
    y_end = layers[i + 1][2][1] + 1.5
    ax.annotate("", xy=(x_arrow, y_end), xytext=(x_arrow, y_start), arrowprops=dict(arrowstyle="->", lw=2))

# Save the architecture diagram
plt.savefig(architecture_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Architecture diagram saved at {architecture_path}")