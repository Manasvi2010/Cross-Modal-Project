import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# ImageNet normalization — must match train_image.py exactly
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])


class FusionFoodDataset(Dataset):
    def __init__(self, root, split, meta_json, vectorizer, class_map):

        self.samples = []
        self.class_map = class_map
        self.vectorizer = vectorizer
        self.transform = TRANSFORM

        with open(meta_json, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        split_dir = os.path.join(root, split)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        missing_meta = []
        missing_map = []

        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)

            if not os.path.isdir(cls_path):
                continue

            if cls not in self.meta:
                missing_meta.append(cls)
                continue

            if cls not in self.class_map:
                missing_map.append(cls)
                continue

            text = self.meta[cls]
            label = self.class_map[cls]

            for img_file in os.listdir(cls_path):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                img_path = os.path.join(cls_path, img_file)
                self.samples.append((img_path, text, label))

        if missing_meta:
            print(f"⚠️  Classes missing in metadata ({len(missing_meta)}): {missing_meta[:5]}...")
        if missing_map:
            print(f"⚠️  Classes missing in class_map ({len(missing_map)}): {missing_map[:5]}...")

        print(f"✅ [{split}] Loaded {len(self.samples)} samples across {len(set(s[2] for s in self.samples))} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"⚠️  Skipping corrupt image: {img_path} — {e}")
            # Return a blank tensor so the batch doesn't crash
            image = torch.zeros(3, 224, 224)

        text_vec = self.vectorizer.transform([text]).toarray()[0]
        text_vec = torch.tensor(text_vec, dtype=torch.float32)

        return image, text_vec, label