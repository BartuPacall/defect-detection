from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, models, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
to_t = transforms.ToTensor()

base = datasets.ImageFolder("data/validation/images")
classes = base.classes

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load("models/resnet18_aug_seed0.pt", map_location=device))
model = model.to(device).eval()

store = {}


def fwd_hook(module, inp, out):
    store["act"] = out
    out.register_hook(lambda g: store.__setitem__("grad", g))


model.layer4.register_forward_hook(fwd_hook)


def gradcam(img):
    x = norm(to_t(img)).unsqueeze(0).to(device)
    model.zero_grad()
    out = model(x)
    pred = out.argmax(1).item()
    out[0, pred].backward()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.size[::-1], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, pred


fig, axes = plt.subplots(6, 4, figsize=(8, 12))
for c in range(6):
    for k, offset in enumerate([5, 35]):
        path, label = base.samples[c * 60 + offset]
        img = Image.open(path).convert("RGB")
        cam, pred = gradcam(img)
        axes[c, 2 * k].imshow(img)
        axes[c, 2 * k + 1].imshow(img)
        axes[c, 2 * k + 1].imshow(cam, cmap="jet", alpha=0.45)
        axes[c, 2 * k].set_title(f"gercek: {classes[label]}", fontsize=7)
        axes[c, 2 * k + 1].set_title(f"tahmin: {classes[pred]}", fontsize=7)
        axes[c, 2 * k].axis("off")
        axes[c, 2 * k + 1].axis("off")

plt.tight_layout()
Path("reports").mkdir(exist_ok=True)
plt.savefig("reports/gradcam.png", dpi=120)
print("Kaydedildi: reports/gradcam.png")