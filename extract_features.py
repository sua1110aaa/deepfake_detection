import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

# 1. 사전 학습된 ResNet18 백본 로드 (비디오 & 오디오 스펙트로그램 공용 특징 추출기)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone.fc = torch.nn.Identity()  # 512차원 Feature Vector 추출용으로 변환
backbone = backbone.to(device)
backbone.eval()

# 이미지/스펙트로그램 정규화 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_real_features(video_path):
    # --- [1. 비디오 특징 추출 (16 프레임 샘플링 + ResNet18)] ---
    v_feat = np.zeros(512, dtype=np.float32)
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor_img = transform(pil_img)
            frames.append(tensor_img)
            
            if len(frames) >= 16:  # 프레임 수 16개로 늘려 공간 특징 강화
                break
        cap.release()

        if frames:
            batch_tensors = torch.stack(frames).to(device)
            with torch.no_grad():
                feats = backbone(batch_tensors)  # [16, 512]
                v_feat = feats.mean(dim=0).cpu().numpy()  # [512]
    except Exception:
        pass

    # --- [2. 오디오 특징 추출 (Mel-Spectrogram 2D 이미지화 + ResNet18)] ---
    a_feat = np.zeros(512, dtype=np.float32)
    try:
        y, sr = librosa.load(video_path, sr=16000, duration=3)
        if len(y) > 0:
            # Mel-Spectrogram 생성
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # 2D Spectrogram을 RGB 3채널 이미지 형태로 변환
            img = Image.fromarray(S_dB).convert('RGB')
            tensor_img = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                a_feat = backbone(tensor_img).squeeze().cpu().numpy()  # [512]
    except Exception:
        pass

    return v_feat, a_feat

def process_csv(csv_path, output_npz_path):
    if not os.path.exists(csv_path):
        print(f"경고: {csv_path} 경로가 존재하지 않습니다.")
        return

    df = pd.read_csv(csv_path)
    v_list, a_list, l_list = [], [], []

    print(f"\n[{csv_path}] ResNet18 + Mel-Spectrogram 고정밀 특징 추출 중... (총 {len(df)}개)")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        v_feat, a_feat = extract_real_features(row['filepath'])
        v_list.append(v_feat)
        a_list.append(a_feat)
        l_list.append(int(row['label']))

    os.makedirs("results", exist_ok=True)
    np.savez(
        output_npz_path,
        video_features=np.array(v_list, dtype=np.float32),
        audio_features=np.array(a_list, dtype=np.float32),
        labels=np.array(l_list, dtype=np.int64)
    )
    print(f" -> 성공적으로 저장됨: {output_npz_path}")

if __name__ == "__main__":
    process_csv("data/train.csv", "results/train_features.npz")
    process_csv("data/valid.csv", "results/valid_features.npz")
    process_csv("data/test.csv", "results/test_features.npz")