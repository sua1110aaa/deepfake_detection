import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from tqdm import tqdm

from src.dataset import DeepfakeEndToEndDataset
from src.multimodal import GMUModel

# End-to-End 모델 구조 정의 (train_e2e.py와 동일)
class EndToEndGMU(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_backbone = models.resnet18(weights=None)
        self.v_backbone.fc = nn.Identity()
        
        self.a_backbone = models.resnet18(weights=None)
        self.a_backbone.fc = nn.Identity()

        self.gmu = GMUModel(video_dim=512, audio_dim=512, hidden_dim=256, num_classes=2)

    def forward(self, v_frames, a_img):
        b, t, c, h, w = v_frames.shape
        v_flat = v_frames.view(b * t, c, h, w)
        v_feats = self.v_backbone(v_flat)
        v_feats = v_feats.view(b, t, 512).mean(dim=1)

        a_feats = self.a_backbone(a_img)

        return self.gmu(v_feats, a_feats)

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"평가 장치: {device}")

    # Test 데이터셋 로드
    test_ds = DeepfakeEndToEndDataset("data/test.csv", num_frames=8)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    # 모델 가중치 로드
    model = EndToEndGMU().to(device)
    model.load_state_dict(torch.load("models/gmu_multimodal.pth", map_location=device))
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    print("\nTest 데이터 평가 중...")
    with torch.no_grad():
        for v, a, l in tqdm(test_loader):
            v, a, l = v.to(device), a.to(device), l.to(device)
            
            outputs = model(v, a)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(l.cpu().numpy())

    # 평가 지표 계산
    acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], digits=2)

    print("\n=================== [최종 평가 결과] ===================")
    print(f"Test Accuracy : {acc:.2f}%")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\n[Confusion Matrix]")
    print(cm)
    print("\n[Classification Report]")
    print(report)

if __name__ == "__main__":
    evaluate()