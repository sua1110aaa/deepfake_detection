import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import DeepfakeEndToEndDataset
from src.multimodal import GMUModel

class EndToEndGMU(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.v_backbone.fc = nn.Identity()
        
        self.a_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.a_backbone.fc = nn.Identity()

        # 백본 layer4 상위층만 파인튜닝
        for name, param in self.v_backbone.named_parameters():
            if "layer4" not in name: param.requires_grad = False
        for name, param in self.a_backbone.named_parameters():
            if "layer4" not in name: param.requires_grad = False

        self.gmu = GMUModel(video_dim=512, audio_dim=512, hidden_dim=256, num_classes=2)

    def forward(self, v_frames, a_img):
        b, t, c, h, w = v_frames.shape
        v_flat = v_frames.view(b * t, c, h, w)
        v_feats = self.v_backbone(v_flat)
        v_feats = v_feats.view(b, t, 512).mean(dim=1)

        a_feats = self.a_backbone(a_img)

        return self.gmu(v_feats, a_feats)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 데이터 로더 설정 (MPS 병목 방지를 위해 batch_size=16, num_workers=0 적용)
    train_ds = DeepfakeEndToEndDataset("data/train.csv", num_frames=8)
    valid_ds = DeepfakeEndToEndDataset("data/valid.csv", num_frames=8)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=0)

    model = EndToEndGMU().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.0]).to(device))

    optimizer = torch.optim.AdamW([
        {'params': model.v_backbone.parameters(), 'lr': 1e-5},
        {'params': model.a_backbone.parameters(), 'lr': 1e-5},
        {'params': model.gmu.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        
        # 진행 상황 출력을 위한 tqdm wrapper 적용
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/10]")
        for v, a, l in pbar:
            v, a, l = v.to(device), a.to(device), l.to(device)

            optimizer.zero_grad()
            out = model(v, a)
            loss = criterion(out, l)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for v, a, l in valid_loader:
                v, a, l = v.to(device), a.to(device), l.to(device)
                out = model(v, a)
                correct += (out.argmax(dim=1) == l).sum().item()
                total += l.size(0)

        acc = (correct / total) * 100
        print(f"Epoch [{epoch+1:02d}/10] | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%\n")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/gmu_multimodal.pth")

if __name__ == "__main__":
    train()