import torch
import torch.nn as nn
from torchvision import models


class ImageModel(nn.Module):

    def __init__(self, num_classes=None):
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Feature extractor — removes the final FC layer
        self.features = nn.Sequential(*list(base.children())[:-1])

        # Classifier — only used during standalone image training
        if num_classes is not None:
            self.classifier = nn.Linear(512, num_classes)
        else:
            self.classifier = None

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        if self.classifier is not None:
            return self.classifier(x)

        return x  # 512-d feature vector


def get_feature_extractor():
    model = ImageModel(num_classes=None)
    return model