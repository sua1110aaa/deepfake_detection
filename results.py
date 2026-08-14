import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from src.multimodal import GMUModel

def evaluate():
    os.makedirs("results", exist_ok=True)
    
    data = np.load("results/test_features.npz")
    v_test = torch.tensor(data['video_features'], dtype=torch.float32)
    a_test = torch.tensor(data['audio_features'], dtype=torch.float32)
    y_true = data['labels']

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = GMUModel(video_dim=512, audio_dim=512, hidden_dim=256, num_classes=2).to(device)
    model.load_state_dict(torch.load("models/gmu_multimodal.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        v_test, a_test = v_test.to(device), a_test.to(device)
        logits = model(v_test, a_test)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, target_names=["Real", "Fake"])

    print("\n=================== [최종 평가 결과] ===================")
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\n[Confusion Matrix]")
    print(cm)
    print("\n[Classification Report]")
    print(report)

    with open("results/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Test ROC-AUC: {auc:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n{report}")

if __name__ == "__main__":
    evaluate()