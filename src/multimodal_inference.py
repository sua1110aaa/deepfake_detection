from pathlib import Path
import torch
import numpy as np

from .inference import VideoDetector
from .multimodal import GatedMultimodalUnit
from .audio import extract_and_clean_audio, wav_to_spectrogram


class MultimodalDeepfakeDetector:
    def __init__(self, video_model_path, gmu_model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. 영상 모델 로드
        self.video_detector = VideoDetector(video_model_path, device=self.device)

        # 2. GMU 융합 모델 로드
        self.gmu_model = GatedMultimodalUnit(v_dim=512, a_dim=512).to(self.device)
        gmu_checkpoint = torch.load(gmu_model_path, map_location=self.device)
        self.gmu_model.load_state_dict(gmu_checkpoint)
        self.gmu_model.eval()

    def predict_video_multimodal(self, video_path):
        # 1. Video Feature 추출
        video_res = self.video_detector.predict_video(video_path, return_features=True)
        v_feat = torch.tensor(video_res["feature"], dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. Audio Feature 추출 (예시: 파이프라인 연동용 512D dummy/spec feature)
        # 추후 AASIST/Audio 백본 모델 연결 시 해당 Feature로 교체
        a_feat = torch.zeros((1, 512), dtype=torch.float32).to(self.device)

        # 3. GMU Gated Fusion 추론
        with torch.no_grad():
            output = self.gmu_model(v_feat, a_feat)
            probs = torch.softmax(output, dim=1)
            fake_prob = probs[0, 1].item()

        return {
            "engine": "GMU-Multimodal",
            "label": "fake" if fake_prob >= 0.5 else "real",
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
            "video_frames_analyzed": video_res["frames_analyzed"]
        }