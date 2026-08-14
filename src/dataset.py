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