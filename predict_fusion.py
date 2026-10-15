import torch
import json
import joblib
import sys
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from image_model import ImageModel
from fusion_model import FusionModel

# ── PATHS ──
IMAGE_MODEL_PATH = "../trained_models/food_image_model.pth"
FUSION_MODEL_PATH = "../trained_models/fusion_model.pth"
VECTORIZER_PATH = "../trained_models/tfidf_vectorizer.pkl"
CLASS_MAP_PATH = "../data_store/food_meta/class_map.json"
META_PATH = "../data_store/food_meta/food_metadata.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── LOAD ──
with open(CLASS_MAP_PATH, "r") as f:
    class_map = json.load(f)

idx_to_class = {v: k for k, v in class_map.items()}
num_classes = len(class_map)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

vectorizer = joblib.load(VECTORIZER_PATH)
text_dim = len(vectorizer.get_feature_names_out())

# ── helper: always returns a plain string for the vectorizer ──
def get_text(class_name):
    entry = metadata.get(class_name, "indian food")
    if isinstance(entry, dict):
        return entry.get("description", "indian food")
    return entry   # already a plain string (old format)

def get_meta(class_name):
    entry = metadata.get(class_name, {})
    if isinstance(entry, dict):
        return entry
    # old plain-string format — wrap it
    return {"description": entry, "calories_per_100g": None,
            "health_tag": "", "spice_level": "", "diet_type": "",
            "origin": "India", "meal_type": "", "fun_fact": ""}

# ── LOAD MODELS ──
image_model = ImageModel(num_classes=None).to(DEVICE)
state_dict = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
filtered = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
image_model.load_state_dict(filtered, strict=False)
image_model.eval()

fusion_model = FusionModel(num_classes=num_classes, text_dim=text_dim).to(DEVICE)
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
fusion_model.eval()

# ── IMAGE ──
if len(sys.argv) < 2:
    print("Usage: python predict_fusion.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ── IMAGE FEATURE ──
with torch.no_grad():
    img_feat = image_model(img_tensor)
    img_feat = F.normalize(img_feat, dim=1)

# ── MULTI-PASS: score each class using its real description text ──
all_scores = []

with torch.no_grad():
    for class_name, class_idx in class_map.items():
        text = get_text(class_name)          # always a plain string now
        text_vec = vectorizer.transform([text]).toarray()
        text_tensor = torch.tensor(text_vec, dtype=torch.float32).to(DEVICE)

        output = fusion_model(img_feat, text_tensor)
        score = output[0][class_idx].item()
        all_scores.append((class_idx, class_name, score))

all_scores.sort(key=lambda x: x[2], reverse=True)

# ── RESULTS ──
print("\n--- TOP 5 PREDICTIONS ---")
for rank, (idx, name, score) in enumerate(all_scores[:5], 1):
    m = get_meta(name)
    label = name.replace("_", " ").title()
    cal = f"  {m['calories_per_100g']} kcal/100g" if m.get("calories_per_100g") else ""
    spice = f"  Spice: {m['spice_level']}" if m.get("spice_level") else ""
    print(f"  {rank}. {label:<35} score: {score:.4f}{cal}{spice}")

top_idx, top_name, top_score = all_scores[0]
m = get_meta(top_name)

print(f"\n🍽️  Predicted:    {top_name.replace('_',' ').title()}")
print(f"📍  Origin:       {m.get('origin', 'India')}")
print(f"🔥  Calories:     {m.get('calories_per_100g', 'N/A')} kcal per 100g")
print(f"💚  Health:       {m.get('health_tag', 'N/A')}")
print(f"🌶️   Spice level:  {m.get('spice_level', 'N/A')}")
print(f"🥛  Diet type:    {m.get('diet_type', 'N/A')}")
print(f"🍽️   Meal type:    {m.get('meal_type', 'N/A')}")
print(f"📝  Description:  {m.get('description', '')}")
if m.get("fun_fact"):
    print(f"✦   Fun fact:     {m.get('fun_fact')}")