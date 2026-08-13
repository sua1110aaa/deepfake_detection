from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []

        # Real = 0
        # Fake = 1
        for label, class_name in enumerate(["real", "fake"]):
            class_dir = self.root_dir / class_name

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label