import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import scipy.sparse

# Set dataset path
dataset_path = "D:/Miniproject/enhanced_Database/train"
img_size = (224, 224)
batch_size = 16  # Reduce batch size to prevent RAM overflow

# Load InceptionV3 (without top layers)
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
gap = GlobalAveragePooling2D()(base_model.output)  # Reduce feature size
model = Model(inputs=base_model.input, outputs=gap)  # Final feature extractor

# Data Generator
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    dataset_path, target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
)

# Extract features
features = model.predict(generator)  # (num_samples, feature_size)
labels = generator.classes  # Labels (0 = NORMAL, 1 = PNEUMONIA)

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to sparse matrix for memory efficiency
X_train_sparse = scipy.sparse.csr_matrix(X_train)
X_test_sparse = scipy.sparse.csr_matrix(X_test)

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    tree_method='hist',  # Use efficient tree method
    max_depth=5,         # Reduce tree depth to save memory
    max_bin=256,         # Reduce bin size
)

xgb_model.fit(X_train_sparse, y_train)

# Predictions
y_pred = xgb_model.predict(X_test_sparse)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
