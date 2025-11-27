# ================================================================
# caption_model.py — Final Combined Model (Classifier + Captioning)
# ================================================================
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ================================================================
# 🔹 VGG16 + GRU Classifier (Jewelry Type)
# ================================================================
class VGG16_GRU_Classifier(nn.Module):
    """Classifies jewelry image as Necklace, Earring, etc."""
    def __init__(self, num_classes=2, hidden_size=256, gru_layers=1):
        super(VGG16_GRU_Classifier, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features

        # Freeze convolutional layers
        for param in self.features.parameters():
            param.requires_grad = False

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (B, h*w, 512)
        _, hidden = self.gru(x)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)


# ================================================================
# 🔹 Encoder (Captioning)
# ================================================================
class EncoderCNN(nn.Module):
    """Extracts local & global visual features using VGG16."""
    def __init__(self, hidden_size):
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, hidden_size)

    def forward(self, images):
        x = self.features(images)        # (B, 512, H, W)
        x = self.pool(x)                 # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 512)
        return self.fc(x)                # (B, hidden_size)


# ================================================================
# 🔹 Decoder (Captioning)
# ================================================================
class DecoderRNN(nn.Module):
    """Generates captions word-by-word."""
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embedded = self.embed(captions)
        hidden = features.unsqueeze(0)  # (1, B, hidden_size)
        outputs, _ = self.gru(embedded, hidden)
        return self.fc(outputs)


# ================================================================
# 🔹 Shared Transform
# ================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ================================================================
# 🔹 Caption Generation (Greedy Decoding)
# ================================================================
def generate_caption(image, encoder, decoder, vocab, device, max_length=30):
    """
    Generate a clean, accurate caption with correct punctuation and formatting.
    Matches notebook decoding but removes leading commas and awkward spaces.
    """
    torch.manual_seed(0)
    encoder.eval()
    decoder.eval()

    # --- Step 1: Preprocess image ---
    image_tensor = transform(image).unsqueeze(0).to(device)

    # --- Step 2: Extract features ---
    with torch.no_grad():
        features = encoder(image_tensor)

    # --- Step 3: Initialize tokens ---
    start_token = vocab.stoi["<START>"]
    end_token = vocab.stoi["<END>"]
    caption = [start_token]

    # --- Step 4: Greedy decoding ---
    with torch.no_grad():
        for _ in range(max_length):
            inputs = torch.tensor(caption).unsqueeze(0).to(device)
            outputs = decoder(features, inputs)
            predicted = outputs.argmax(2)[:, -1].item()
            caption.append(predicted)
            if predicted == end_token:
                break

    # --- Step 5: Convert token IDs → words (skip unwanted tokens) ---
    words = []
    for idx in caption:
        word = vocab.itos.get(idx, "")
        if word not in ["<PAD>", "<START>", "<END>", "<UNK>", ""]:
            words.append(word)

    # --- Step 6: Remove leading punctuation and fix spacing ---
    # If first token is a punctuation mark, drop it
    while words and words[0] in [",", ".", "!", "?", ";", ":"]:
        words.pop(0)

    sentence = " ".join(words)

    # --- Step 7: Clean spacing and punctuation ---
    sentence = (
        sentence.replace(" ,", ",")
                .replace(" .", ".")
                .replace(" !", "!")
                .replace(" ?", "?")
                .replace(" ;", ";")
                .replace(" :", ":")
                .replace(" 's", "'s")
                .strip()
    )

    # --- Step 8: Capitalize and ensure period ---
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            sentence += "."

    return sentence

