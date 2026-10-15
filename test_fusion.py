import os
import json
import torch
import joblib
from torch.utils.data import DataLoader

from image_model import ImageModel
from fusion_model import FusionModel
from fusion_dataset import FusionFoodDataset

import torch.nn.functional as F


def main():
    # -------- DEVICE --------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # -------- PATHS --------
    IMAGE_ROOT = "../data_store/food_images"
    META_JSON = "../data_store/food_meta/food_metadata.json"
    CLASS_MAP_JSON = "../data_store/food_meta/class_map.json"

    IMAGE_MODEL_PATH = "../trained_models/food_image_model.pth"
    FUSION_MODEL_PATH = "../trained_models/fusion_model.pth"
    VECTORIZER_PATH = "../trained_models/tfidf_vectorizer.pkl"

    BATCH_SIZE = 32

    # -------- LOAD --------
    with open(CLASS_MAP_JSON, "r") as f:
        class_map = json.load(f)

    vectorizer = joblib.load(VECTORIZER_PATH)
    text_dim = len(vectorizer.get_feature_names_out())
    print(f"Text feature dim: {text_dim}")

    # -------- DATASET --------
    test_dataset = FusionFoodDataset(
        root=IMAGE_ROOT,
        split="valid",
        meta_json=META_JSON,
        vectorizer=vectorizer,
        class_map=class_map
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -------- LOAD MODELS --------
    image_model = ImageModel(num_classes=None).to(DEVICE)

    state_dict = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
    image_model.load_state_dict(filtered, strict=False)
    image_model.eval()

    fusion_model = FusionModel(num_classes=len(class_map), text_dim=text_dim).to(DEVICE)
    fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
    fusion_model.eval()

    print("✅ Models loaded")

    # -------- EVALUATE --------
    correct = 0
    total = 0

    print("\nRunning evaluation...")

    with torch.no_grad():
        for images, texts_vec, labels in test_loader:
            images = images.to(DEVICE)
            texts_vec = texts_vec.to(DEVICE)
            labels = labels.to(DEVICE)

            img_feat = image_model(images)
            img_feat = F.normalize(img_feat, dim=1)

            outputs = fusion_model(img_feat, texts_vec)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(f"\n✅ Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct} / {total}")


if __name__ == "__main__":
    main()