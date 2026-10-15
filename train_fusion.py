import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from image_model import ImageModel
from fusion_model import FusionModel
from fusion_dataset import FusionFoodDataset


def main():

    # -------- DEVICE --------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # -------- PATHS --------
    IMAGE_ROOT = "../data_store/food_images"
    META_JSON = "../data_store/food_meta/food_metadata.json"
    CLASS_MAP_PATH = "../data_store/food_meta/class_map.json"

    IMAGE_MODEL_PATH = "../trained_models/food_image_model.pth"
    FUSION_MODEL_SAVE = "../trained_models/fusion_model.pth"
    VECTORIZER_SAVE = "../trained_models/tfidf_vectorizer.pkl"

    # -------- HYPERPARAMS --------
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 3e-4

    # -------- LOAD CLASS MAP --------
    with open(CLASS_MAP_PATH, "r") as f:
        class_map = json.load(f)

    num_classes = len(class_map)
    print(f"Number of classes: {num_classes}")

    # -------- LOAD METADATA --------
    with open(META_JSON, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    texts = list(metadata.values())

    # -------- TF-IDF --------
    print("⚙️ Building TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    vectorizer.fit(texts)
    joblib.dump(vectorizer, VECTORIZER_SAVE)

    text_dim = len(vectorizer.get_feature_names_out())
    print(f"✅ TF-IDF vectorizer saved | Text feature size: {text_dim}")

    # -------- DATASETS --------
    train_dataset = FusionFoodDataset(
        root=IMAGE_ROOT,
        split="train",
        meta_json=META_JSON,
        vectorizer=vectorizer,
        class_map=class_map
    )

    val_dataset = FusionFoodDataset(
        root=IMAGE_ROOT,
        split="valid",
        meta_json=META_JSON,
        vectorizer=vectorizer,
        class_map=class_map
    )

    # IMPORTANT: num_workers = 0 for Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # -------- IMAGE MODEL --------
    print("📷 Loading image model...")

    image_model = ImageModel(num_classes=None).to(DEVICE)

    state_dict = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
    filtered = {k: v for k, v in state_dict.items()
                if not k.startswith("classifier")}

    image_model.load_state_dict(filtered, strict=False)

    image_model.eval()
    for p in image_model.parameters():
        p.requires_grad = False

    print("✅ Image model loaded and frozen (512-d features)")

    # -------- FUSION MODEL --------
    fusion_model = FusionModel(
        num_classes=num_classes,
        text_dim=text_dim
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # -------- TRAIN --------
    print("\n🚀 Starting fusion training...\n")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        fusion_model.train()
        total_loss = 0
        correct = 0
        total = 0

        for imgs, texts_vec, labels in train_loader:
            imgs = imgs.to(DEVICE)
            texts_vec = texts_vec.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                img_features = image_model(imgs)
                img_features = F.normalize(img_features, dim=1)

            outputs = fusion_model(img_features, texts_vec)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # ---- VALIDATE ----
        fusion_model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, texts_vec, labels in val_loader:
                imgs = imgs.to(DEVICE)
                texts_vec = texts_vec.to(DEVICE)
                labels = labels.to(DEVICE)

                img_features = image_model(imgs)
                img_features = F.normalize(img_features, dim=1)

                outputs = fusion_model(img_features, texts_vec)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}]  "
              f"Loss: {total_loss/len(train_loader):.4f}  "
              f"Train: {train_acc:.2f}%  "
              f"Val: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(fusion_model.state_dict(), FUSION_MODEL_SAVE)
            print(f"  💾 Best fusion model saved (Val Acc: {val_acc:.2f}%)")

    print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%")
    print(f"Fusion model saved to: {FUSION_MODEL_SAVE}")
    print(f"Vectorizer saved to:   {VECTORIZER_SAVE}")


if __name__ == "__main__":
    main()