import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class AlbDataset(Dataset):
    def __init__(self, folder, transform):
        self.base = datasets.ImageFolder(folder)
        self.transform = transform
        self.classes = self.base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        path, label = self.base.samples[i]
        img = np.array(Image.open(path).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, label


def get_loaders_aug(batch_size=32):
    train_tfm = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    val_tfm = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

    train_ds = AlbDataset("data/train/images", train_tfm)
    val_ds = AlbDataset("data/validation/images", val_tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, train_ds.classes