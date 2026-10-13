import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import datasets, models, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
to_t = transforms.ToTensor()

base = datasets.ImageFolder("data/validation/images")
imgs = [(Image.open(p).convert("RGB"), y) for p, y in base.samples]


def load(path):
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, 6)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def distort(img, brightness, contrast):
    a = np.asarray(img).astype(np.float32)
    a = (a - 128) * contrast + 128 + brightness
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def accuracy(model, brightness, contrast):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(imgs), 60):
            chunk = imgs[i:i + 60]
            x = torch.stack([norm(to_t(distort(im, brightness, contrast))) for im, _ in chunk]).to(device)
            y = torch.tensor([y for _, y in chunk]).to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(imgs)


settings = [("normal", 0, 1.0), ("parlak +60", 60, 1.0), ("koyu -60", -60, 1.0),
            ("kontrast x0.5", 0, 0.5), ("kontrast x1.5", 0, 1.5)]

for name, path in [("artirmasiz", "models/resnet18_seed{}.pt"), ("artirmali", "models/resnet18_aug_seed{}.pt")]:
    print("==", name, "==")
    models_ = [load(path.format(s)) for s in range(3)]
    for label, b, c in settings:
        accs = [accuracy(m, b, c) for m in models_]
        print(f"{label:14s} ortalama {np.mean(accs):.4f}  (seed'ler: {[round(a, 4) for a in accs]})")