import os
import numpy as np
import cv2
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

# Define dataset path
data_path = "D:/Miniproject/enhanced_Database/"
categories = ["NORMAL", "PNEUMONIA"]

# Function to extract HOG features
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      orientations=9, block_norm='L2-Hys', visualize=True)
    return features

# Load dataset
X, y = [], []
for label, category in enumerate(categories):
    folder_path = os.path.join(data_path, "train", category)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (128, 128))
            features = extract_hog_features(image)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(eval_metric='logloss', tree_method='hist')
model.fit(X_train, y_train)

# Save the model
model_path = "D:/Miniproject/XGBOOST/xgboost_model.pkl"
joblib.dump(model, model_path)

# Generate Feature Importance Plot
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type="weight", max_num_features=10)
plt.title("Feature Importance (XGBoost)")
plt.savefig("D:/Miniproject/XGBOOST/feature_importance.png")
plt.close()

# Generate Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("D:/Miniproject/XGBOOST/confusion_matrix.png")
plt.close()

# Generate ROC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("D:/Miniproject/XGBOOST/roc_curve.png")
plt.close()

# Generate Classification Report Table
report = classification_report(y_test, y_pred, target_names=categories, output_dict=True)
report_text = "Classification Report:\n\n"
for label, metrics in report.items():
    if isinstance(metrics, dict):
        report_text += f"{label}:\n"
        for metric, value in metrics.items():
            report_text += f"  {metric}: {value:.4f}\n"
        report_text += "\n"

with open("classification_report.txt", "w") as f:
    f.write(report_text)

# Generate SHAP Summary Plot
explainer = shap.Explainer(model)
shap_values = explainer(X_test[:100])
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test[:100], show=False)
plt.savefig("D:/Miniproject/XGBOOST/shap_summary.png", bbox_inches="tight")
plt.close()

print("All visualizations saved successfully!")
