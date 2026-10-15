import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from image_model import ImageModel


def main():

    # -------- CONFIG --------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    DATA_DIR = "../data_store/food_images"
    MODEL_PATH = "../trained_models/food_image_model.pth"
    CLASS_MAP_PATH = "../data_store/food_meta/class_map.json"

    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 1e-4

    # -------- TRANSFORMS --------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------- DATASET --------
    train_ds = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=transform
    )

    val_ds = datasets.ImageFolder(
        os.path.join(DATA_DIR, "valid"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0   # IMPORTANT FOR WINDOWS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0   # IMPORTANT FOR WINDOWS
    )

    num_classes = len(train_ds.classes)
    print(f"Number of classes: {num_classes}")

    # -------- SAVE CLASS MAP --------
    os.makedirs("../trained_models", exist_ok=True)
    os.makedirs("../data_store/food_meta", exist_ok=True)

    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    print(f"Class map saved to {CLASS_MAP_PATH}")

    # -------- MODEL --------
    model = ImageModel(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # -------- TRAIN --------
    best_val_acc = 0.0

    for epoch in range(EPOCHS):

        # TRAIN
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total

        # VALIDATE
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {total_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best model saved (Val Acc: {val_acc:.2f}%)")

    print(f"\nImage model training complete. Best Val Acc: {best_val_acc:.2f}%")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()