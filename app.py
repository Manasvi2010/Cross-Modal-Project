"""
Flask API Backend for Indian Food Classifier
Run from project ROOT (INDIANFOOD_CROSSMODAL_V2/):
    python app.py
"""

import os, sys, json, io
import torch, joblib
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))
from image_model import ImageModel
from fusion_model import FusionModel

# ── PATHS ──
IMAGE_MODEL_PATH  = "trained_models/food_image_model.pth"
FUSION_MODEL_PATH = "trained_models/fusion_model.pth"
VECTORIZER_PATH   = "trained_models/tfidf_vectorizer.pkl"
CLASS_MAP_PATH    = "data_store/food_meta/class_map.json"
META_PATH         = "data_store/food_meta/food_metadata.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[server] Using device: {DEVICE}")

# ── LOAD ──
with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)

idx_to_class = {v: k for k, v in class_map.items()}
num_classes  = len(class_map)

with open(META_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

vectorizer = joblib.load(VECTORIZER_PATH)
text_dim   = len(vectorizer.get_feature_names_out())

# image model
image_model = ImageModel(num_classes=None).to(DEVICE)
sd = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
image_model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("classifier")}, strict=False)
image_model.eval()

# fusion model
fusion_model = FusionModel(num_classes=num_classes, text_dim=text_dim).to(DEVICE)
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
fusion_model.eval()

print("[server] ✅ Models loaded")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

app = Flask(__name__)
CORS(app)


def get_meta(class_name):
    """Handles both old plain-string and new rich-dict metadata formats."""
    raw = metadata.get(class_name, {})
    if isinstance(raw, str):
        return {
            "description": raw, "ingredients": [],
            "calories_per_100g": None, "health_tag": "Unknown",
            "spice_level": "Unknown", "diet_type": "Unknown",
            "origin": "India", "meal_type": "Unknown", "fun_fact": ""
        }
    return raw


def predict_image(pil_img):
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_feat = image_model(img_tensor)
        img_feat = F.normalize(img_feat, dim=1)

    scores = []
    with torch.no_grad():
        for class_name, class_idx in class_map.items():
            m    = get_meta(class_name)
            text = m["description"] if isinstance(m, dict) else m
            text_vec = vectorizer.transform([text]).toarray()
            text_t   = torch.tensor(text_vec, dtype=torch.float32).to(DEVICE)
            output   = fusion_model(img_feat, text_t)
            score    = output[0][class_idx].item()
            scores.append((class_name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top5  = scores[:5]
    probs = torch.softmax(torch.tensor([s for _, s in top5]), dim=0).tolist()

    results = []
    for (name, _), prob in zip(top5, probs):
        m = get_meta(name)
        results.append({
            "class":             name,
            "label":             name.replace("_", " ").title(),
            "probability":       round(prob * 100, 2),
            "description":       m.get("description", ""),
            "ingredients":       m.get("ingredients", []),
            "calories_per_100g": m.get("calories_per_100g"),
            "health_tag":        m.get("health_tag", ""),
            "spice_level":       m.get("spice_level", ""),
            "diet_type":         m.get("diet_type", ""),
            "origin":            m.get("origin", ""),
            "meal_type":         m.get("meal_type", ""),
            "fun_fact":          m.get("fun_fact", ""),
        })
    return results


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400
    try:
        results = predict_image(pil_img)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500
    return jsonify({"predictions": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": num_classes, "device": DEVICE})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)