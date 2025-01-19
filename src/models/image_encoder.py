import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, model_name, hidden_size):
        super().__init__()
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, hidden_size)

    def forward(self, x):
        return self.model(x)