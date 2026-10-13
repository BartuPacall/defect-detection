from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(batch_size=32):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_ds = datasets.ImageFolder("data/train/images", transform=tfm)
    val_ds = datasets.ImageFolder("data/validation/images", transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, train_ds.classes