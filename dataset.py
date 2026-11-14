import random
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    DATA_ROOT, MANIFEST_PATH, IMAGE_SIZE,
    CLASS_TO_IDX, NUM_WORKERS, SEED
)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, scale=(0.8, 1.2)),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor()
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


class DFDCDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = DATA_ROOT / row["relative_path"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)
        label = CLASS_TO_IDX[row["label"]]

        return image, torch.tensor(label, dtype=torch.long)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(batch_size):
    df = pd.read_csv(MANIFEST_PATH)

    loaders = {}

    for split in ["train", "valid", "test"]:
        dataset = DFDCDataset(
            df[df["split"] == split],
            train_transform if split == "train" else eval_transform
        )

        generator = torch.Generator().manual_seed(SEED)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator
        )

    return loaders
