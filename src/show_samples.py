from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

root = Path("data/train/images")
classes = sorted(d.name for d in root.iterdir() if d.is_dir())

fig, axes = plt.subplots(len(classes), 4, figsize=(8, 12))
for i, cls in enumerate(classes):
    files = sorted((root / cls).iterdir())[:4]
    for j, f in enumerate(files):
        axes[i, j].imshow(Image.open(f))
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(cls, fontsize=9, loc="left")

plt.tight_layout()
Path("reports").mkdir(exist_ok=True)
plt.savefig("reports/samples.png", dpi=120)
print("Kaydedildi: reports/samples.png")