import torch
import torch.nn as nn
import torch.nn.functional as F

class GMUModel(nn.Module):
    """
    Gated Multimodal Unit (GMU) 기반 딥페이크 탐지 결합 모델
    """
    def __init__(self, video_dim=512, audio_dim=512, hidden_dim=256, num_classes=2):
        super(GMUModel, self).__init__()
        
        # 비디오/오디오 특징 차원 축소 및 변환 레이어
        self.W_v = nn.Linear(video_dim, hidden_dim)
        self.W_a = nn.Linear(audio_dim, hidden_dim)
        
        # Gate 계산용 레이어 (비디오와 오디오 가중치 결정)
        self.W_z = nn.Linear(video_dim + audio_dim, hidden_dim)
        
        # 분류기 (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, v, a):
        # 1. 각 모달리티의 비선형 특징 변환 (tanh activation)
        h_v = torch.tanh(self.W_v(v))
        h_a = torch.tanh(self.W_a(a))
        
        # 2. Gating Vector 계산 (Sigmoid -> 0~1 사이 가중치)
        concat_features = torch.cat([v, a], dim=1)
        z = torch.sigmoid(self.W_z(concat_features))
        
        # 3. Gated Fusion (z * h_v + (1 - z) * h_a)
        h_fusion = z * h_v + (1 - z) * h_a
        
        # 4. 최종 로짓 출력
        out = self.classifier(h_fusion)
        return out
