import torch, torch.nn as nn
from torchvision import models

def build_model(pretrained=True):
    w = models.ResNet50_Weights.DEFAULT if pretrained else None
    m = models.resnet50(weights=w)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 5))
    return m

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
