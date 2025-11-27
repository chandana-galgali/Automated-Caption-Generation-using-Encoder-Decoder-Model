import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pickle
import sys

class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {}
        self.stoi = {}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

# Register class under __main__ so pickle can find it
sys.modules['__main__'].Vocabulary = Vocabulary

# ----- Models from caption_model.py -----
from caption_model import (
    VGG16_GRU_Classifier,
    EncoderCNN,
    DecoderRNN,
    generate_caption,
    transform
)

# ======================================================
# 🔹 Flask Setup
# ======================================================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ======================================================
# 🔹 1. Load Vocab (pickle)
# ======================================================
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
print("Vocab loaded with size:", vocab_size)

# ======================================================
# 🔹 2. Load Jewelry Classifier
# ======================================================
CLASS_NAMES = ["Earring", "Necklace"]

classifier = VGG16_GRU_Classifier(num_classes=len(CLASS_NAMES))
classifier.load_state_dict(torch.load("model.pth", map_location=device))
classifier.to(device)
classifier.eval()

print("Jewelry Type Classifier Loaded!")

# ======================================================
# 🔹 3. Load Caption Models (Encoder + Decoder)
# ======================================================
embed_size = 256
hidden_size = 512

encoder = EncoderCNN(hidden_size).to(device)
encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
encoder.eval()

decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
decoder.eval()

print("Caption Encoder + Decoder Loaded!")

# ======================================================
# 🔹 4. Route: /predict
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # --- Load image ---
    img = Image.open(filepath).convert("RGB")

    # ----------------------------------------------------
    # 🔹 Predict Jewelry Type
    # ----------------------------------------------------
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier(img_tensor)
        _, pred_idx = torch.max(outputs, 1)
        jewelry_type = CLASS_NAMES[pred_idx.item()]

    # ----------------------------------------------------
    # 🔹 Generate Caption
    # ----------------------------------------------------
    caption = generate_caption(img, encoder, decoder, vocab, device)

    # ----------------------------------------------------
    # 🔹 Return Both
    # ----------------------------------------------------
    return jsonify({
        "type": jewelry_type,
        "description": caption
    })


# ======================================================
# 🔹 Start Server
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
