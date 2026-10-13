from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_loaders(batch_size=32, imagenet=False):
    if imagenet:
        norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    else:
        norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    tfm = transforms.Compose([transforms.ToTensor(), norm])
    train_ds = datasets.ImageFolder("data/train/images", transform=tfm)
    val_ds = datasets.ImageFolder("data/validation/images", transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, train_ds.classes