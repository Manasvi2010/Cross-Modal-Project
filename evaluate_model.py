import os
import json
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from image_model import ImageModel
from fusion_model import FusionModel
from fusion_dataset import FusionFoodDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -------- PATHS --------
IMAGE_ROOT = "../data_store/food_images"
META_JSON = "../data_store/food_meta/food_metadata.json"
CLASS_MAP_PATH = "../data_store/food_meta/class_map.json"

IMAGE_MODEL_PATH = "../trained_models/food_image_model.pth"
FUSION_MODEL_PATH = "../trained_models/fusion_model.pth"
VECTORIZER_PATH = "../trained_models/tfidf_vectorizer.pkl"

os.makedirs("../outputs", exist_ok=True)

# -------- LOAD --------
with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)

idx_to_class = {v: k for k, v in class_map.items()}

# CRITICAL: load the same vectorizer from training
vectorizer = joblib.load(VECTORIZER_PATH)
text_dim = len(vectorizer.get_feature_names_out())
print(f"Text feature dim: {text_dim}")

dataset = FusionFoodDataset(
    root=IMAGE_ROOT,
    split="valid",
    meta_json=META_JSON,
    vectorizer=vectorizer,
    class_map=class_map
)

loader = DataLoader(dataset, batch_size=32, shuffle=False,
                    num_workers=2, pin_memory=True)

# -------- LOAD MODELS --------
image_model = ImageModel(num_classes=None).to(DEVICE)
state_dict = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
filtered = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
image_model.load_state_dict(filtered, strict=False)
image_model.eval()

fusion_model = FusionModel(num_classes=len(class_map), text_dim=text_dim).to(DEVICE)
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
fusion_model.eval()

print("✅ Models loaded. Running evaluation...")

# -------- EVALUATE --------
y_true = []
y_pred = []

with torch.no_grad():
    for img, text_vec, label in loader:
        img = img.to(DEVICE)
        text_vec = text_vec.to(DEVICE)

        img_feat = image_model(img)
        img_feat = F.normalize(img_feat, dim=1)

        out = fusion_model(img_feat, text_vec)
        preds = torch.argmax(out, dim=1)

        y_true.extend(label.numpy())
        y_pred.extend(preds.cpu().numpy())

# -------- REPORT --------
class_names = [idx_to_class[i] for i in range(len(class_map))]

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -------- CONFUSION MATRIX --------
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(20, 20))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=16)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)

# Add class labels if not too many
if len(class_names) <= 40:
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("../outputs/confusion_matrix.png", dpi=150)
plt.close()

accuracy = 100 * sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
print(f"\n✅ Overall Accuracy: {accuracy:.2f}%")
print("✅ Confusion matrix saved to outputs/confusion_matrix.png")