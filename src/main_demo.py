from src.multimodal_inference import MultimodalDeepfakeDetector

def main():
    detector = MultimodalDeepfakeDetector(
        video_model_path="models/resnet18_test.pth",
        gmu_model_path="models/gmu_multimodal.pth"
    )

    test_video_path = "data/sample.mp4" # 테스트할 영상 경로
    
    try:
        result = detector.predict_video_multimodal(test_video_path)
        print("\n=== 딥페이크 멀티모달 최종 판정 결과 ===")
        print(f"판정 라벨       : {result['label'].upper()}")
        print(f"Fake 확률       : {result['fake_probability'] * 100:.2f}%")
        print(f"Real 확률       : {result['real_probability'] * 100:.2f}%")
        print(f"분석된 프레임 수 : {result['video_frames_analyzed']}개")
    except Exception as e:
        print(f"추론 실패: {e}")

if __name__ == "__main__":
    main()