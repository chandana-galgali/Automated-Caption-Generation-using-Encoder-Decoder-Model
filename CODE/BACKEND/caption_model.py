import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image



class VGG16_GRU_Classifier(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, gru_layers=1):
        super(VGG16_GRU_Classifier, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features

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
        x = x.view(b, c, h * w).permute(0, 2, 1)
        _, hidden = self.gru(x)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)


class EncoderCNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, hidden_size)

    def forward(self, images):
        x = self.features(images)     
        x = self.pool(x)            
        x = x.view(x.size(0), -1)      
        return self.fc(x)               



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embedded = self.embed(captions)
        hidden = features.unsqueeze(0) 
        outputs, _ = self.gru(embedded, hidden)
        return self.fc(outputs)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def generate_caption(image, encoder, decoder, vocab, device, max_length=30):
    torch.manual_seed(0)
    encoder.eval()
    decoder.eval()

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)

    start_token = vocab.stoi["<START>"]
    end_token = vocab.stoi["<END>"]
    caption = [start_token]

    with torch.no_grad():
        for _ in range(max_length):
            inputs = torch.tensor(caption).unsqueeze(0).to(device)
            outputs = decoder(features, inputs)
            predicted = outputs.argmax(2)[:, -1].item()
            caption.append(predicted)
            if predicted == end_token:
                break

    words = []
    for idx in caption:
        word = vocab.itos.get(idx, "")
        if word not in ["<PAD>", "<START>", "<END>", "<UNK>", ""]:
            words.append(word)

    while words and words[0] in [",", ".", "!", "?", ";", ":"]:
        words.pop(0)

    sentence = " ".join(words)

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

    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            sentence += "."

    return sentence


