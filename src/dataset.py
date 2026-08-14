from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        csv_file 구조 예시:
        filepath,label
        data/real/001.jpg,0
        data/fake/002.jpg,1
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = Path(row['filepath'])
        label = int(row['label'])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import librosa

class DeepfakeEndToEndDataset(Dataset):
    def __init__(self, csv_path, num_frames=8):
        self.df = pd.read_csv(csv_path)
        self.num_frames = num_frames
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['filepath']
        label = int(row['label'])

        # 1. 비디오 프레임 로드 및 샘플링
        v_tensor = torch.zeros(self.num_frames, 3, 224, 224)
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(Image.fromarray(rgb)))
                if len(frames) >= self.num_frames: break
            cap.release()

            if frames:
                v_tensor[:len(frames)] = torch.stack(frames)
        except Exception:
            pass

        # 2. 오디오 Mel-Spectrogram 로드 및 이미지화
        a_tensor = torch.zeros(3, 224, 224)
        try:
            y, sr = librosa.load(video_path, sr=16000, duration=3)
            if len(y) > 0:
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = Image.fromarray(S_dB).convert('RGB')
                a_tensor = self.transform(img)
        except Exception:
            pass

        return v_tensor, a_tensor, label