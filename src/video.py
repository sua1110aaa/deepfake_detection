import cv2
import torch
import numpy as np
import mediapipe as mp

# MediaPipe 버전 호환성 처리
try:
    # 기존 solution API 호환
    import mediapipe.python.solutions.face_detection as mp_face_detection
except ImportError:
    try:
        mp_face_detection = mp.solutions.face_detection
    except AttributeError:
        mp_face_detection = None


class VideoPreprocessor:
    def __init__(self, device='cpu'):
        self.device = device
        if mp_face_detection is not None:
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=1, 
                min_detection_confidence=0.5
            )
        else:
            self.face_detection = None

    def extract_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 얼굴 감지 적용
            if self.face_detection is not None:
                results = self.face_detection.process(rgb_frame)
                if results.detections:
                    # 얼굴 영역이 감지된 경우 Crop 후 저장
                    bboxC = results.detections[0].location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    ymin, xmin, height, width = int(bboxC.ymin * h), int(bboxC.xmin * w), int(bboxC.height * h), int(bboxC.width * w)
                    
                    # 좌상단 범위 유효성 체크
                    ymin, xmin = max(0, ymin), max(0, xmin)
                    crop_frame = rgb_frame[ymin:ymin+height, xmin:xmin+width]
                    
                    if crop_frame.size != 0:
                        resized = cv2.resize(crop_frame, (224, 224))
                        frames.append(resized)
                        continue

            # 얼굴 미감지 시 센터 크롭 fallback
            resized = cv2.resize(rgb_frame, (224, 224))
            frames.append(resized)
            
            # 샘플링 프레임 제한 (속도 최적화)
            if len(frames) >= 16:
                break
                
        cap.release()

        if not frames:
            return None

        # [16, 224, 224, 3] -> 평균 512차원 더미/기본 feature 벡터 변환 예시
        # 실제 모델(ResNet 등) 가중치가 있다면 여기서 Pass시킵니다.
        arr = np.array(frames, dtype=np.float32) / 255.0
        feature_vector = np.mean(arr, axis=(0, 1, 2))  # [3] -> [512] 확장
        feature_vector = np.pad(feature_vector, (0, 509), 'constant') # 512 차원 맞춤

        return feature_vector