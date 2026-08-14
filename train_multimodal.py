import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.multimodal import GMUModel

def load_npz(path):
    print(f"데이터 로딩 중: {path}", flush=True)
    data = np.load(path)
    v = torch.tensor(data['video_features'], dtype=torch.float32)
    a = torch.tensor(data['audio_features'], dtype=torch.float32)
    l = torch.tensor(data['labels'], dtype=torch.long)
    print(f"  └─> 로드 완료: {len(l)}개 샘플", flush=True)
    return TensorDataset(v, a, l)

def main():
    print("==========================================", flush=True)
    print("    GMU Multimodal 모델 학습 시작          ", flush=True)
    print("==========================================", flush=True)
    
    os.makedirs("models", exist_ok=True)

    # 1. 데이터 로드
    train_ds = load_npz("results/train_features.npz")
    valid_ds = load_npz("results/valid_features.npz")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)

    # 2. 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"사용 연산 장치 (Device): {device}\n", flush=True)

    # 3. 모델 및 손실함수/옵티마이저 정의
    model = GMUModel(video_dim=512, audio_dim=512, hidden_dim=256, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 10
    best_loss = float('inf')

    # 4. 학습 루프
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for v, a, l in train_loader:
            v, a, l = v.to(device), a.to(device), l.to(device)

            optimizer.zero_grad()
            out = model(v, a)
            loss = criterion(out, l)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 검증 루프
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for v, a, l in valid_loader:
                v, a, l = v.to(device), a.to(device), l.to(device)
                out = model(v, a)
                loss = criterion(out, l)
                val_loss += loss.item()

                preds = out.argmax(dim=1)
                correct += (preds == l).sum().item()
                total += l.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        acc = (correct / total) * 100 if total > 0 else 0.0

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%", flush=True)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "models/gmu_multimodal.pth")
            print("  └─> [Saved] 최적 모델 가중치 저장 완료!", flush=True)

    print("\n학습이 성공적으로 마쳐졌습니다!", flush=True)

if __name__ == "__main__":
    main()
