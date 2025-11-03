import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")  # Suppress any warnings

# Define path to feature files
feature_path = "D:/Miniproject/InceptionV3/Features"

# Define classification save path
classification_path = "D:/Miniproject/InceptionV3/Classifications/SVM"
os.makedirs(classification_path, exist_ok=True)

# Load train, validation, and test feature files
train_data = np.load(os.path.join(feature_path, "train_features.npz"))
val_data = np.load(os.path.join(feature_path, "val_features.npz"))
test_data = np.load(os.path.join(feature_path, "test_features.npz"))

# Concatenate train and validation sets
X_train = np.vstack((train_data["features"], val_data["features"]))
y_train = np.hstack((train_data["labels"], val_data["labels"]))

# Test set
X_test = test_data["features"]
y_test = test_data["labels"]

# Scale features since SVMs are sensitive to feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define SVM classifier (using RBF kernel for non-linearity)
svm_model = SVC(kernel="rbf", probability=True, random_state=42)

# Train the SVM classifier
print("Training SVM model...")
svm_model.fit(X_train_scaled, y_train)
print("SVM training complete!")

# Make predictions and compute probabilities
y_pred = svm_model.predict(X_test_scaled)
y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("SVM Classification Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot and save the confusion matrix
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (SVM)")
plt.savefig(os.path.join(classification_path, "svm_confusion_matrix.png"), dpi=300)
plt.close()

# Plot and save the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (SVM)")
plt.legend(loc="lower right")
plt.savefig(os.path.join(classification_path, "svm_roc_curve.png"), dpi=300)
plt.close()

print("SVM Classification & Evaluation complete!")
