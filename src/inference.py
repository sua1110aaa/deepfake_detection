import numpy as np
import torch
from PIL import Image

from .model import DeepfakeDetector
from .preprocessing import get_inference_transform
from .video import extract_faces_from_video

class VideoDetector:
    def __init__(self, model_path, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = DeepfakeDetector(num_classes=2)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = get_inference_transform()

    def predict_video(self, video_path, sample_rate=10, return_features=False):
        # 1. Face Crop된 프레임 extraction
        faces = extract_faces_from_video(video_path, sample_rate=sample_rate)

        fake_scores = []
        features_list = []

        for face in faces:
            image = Image.fromarray(face)
            x = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if return_features:
                    feat = self.model(x, return_feature=True)
                    features_list.append(feat.cpu().numpy())
                
                output = self.model(x)
                probabilities = torch.softmax(output, dim=1)
                fake_scores.append(probabilities[0, 1].item())

        fake_probability = float(np.mean(fake_scores))
        real_probability = 1.0 - fake_probability

        result = {
            "engine": "video",
            "label": "fake" if fake_probability >= 0.5 else "real",
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "frames_analyzed": len(faces)
        }

        # GMU 결합을 위해 평균 512차원 Feature Vector 반환
        if return_features:
            avg_feature = np.mean(np.concatenate(features_list, axis=0), axis=0)
            result["feature"] = avg_feature

        return result