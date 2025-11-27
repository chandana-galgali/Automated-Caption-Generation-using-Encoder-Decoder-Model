import torch
import torch.nn as nn
from torchvision import models

class VGG16_GRU_Classifier(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, gru_layers=1):
        super(VGG16_GRU_Classifier, self).__init__()

        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = vgg16.features

        # Freeze feature extractor
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
        batch_size, channels, h, w = x.size()
        x = x.view(batch_size, channels, h * w)
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)
        out = self.fc(h_n)
        return out


def predict_image(model, image, device, transform):
    model.eval()
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()
