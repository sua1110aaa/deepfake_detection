import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights="DEFAULT")
        
        # 마지막 fc layer 제거 및 차원 정보 저장
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        
        # 새로운 Classifier
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x, return_feature=False):
        # 512차원 Feature 추출
        feature = self.backbone(x) 
        
        if return_feature:
            return feature
            
        out = self.fc(feature)
        return out