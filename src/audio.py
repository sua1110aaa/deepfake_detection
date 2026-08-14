import torch
import numpy as np
import librosa


class AudioPreprocessor:
    def __init__(self, sample_rate=16000, duration=3):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration

    def extract_features(self, video_path):
        try:
            # librosa를 사용해 비디오 파일에서 오디오 직접 로드
            y, sr = librosa.load(video_path, sr=self.sample_rate, duration=self.duration)

            # 길이 맞추기 (Padding 또는 Crop)
            if len(y) < self.target_length:
                y = np.pad(y, (0, self.target_length - len(y)), mode='constant')
            else:
                y = y[:self.target_length]

            # MFCC 특징 추출 (13개 계수)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0) # [13]

            # 512 차원 벡터로 핑 맞춤 (더미/기본)
            feature_vector = np.pad(mfcc_mean, (0, 512 - len(mfcc_mean)), mode='constant')

            return feature_vector
        except Exception:
            # 오디오 트랙이 없는 영상 등의 경우 Fallback 0 벡터 반환
            return np.zeros(512, dtype=np.float32)