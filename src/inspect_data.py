from pathlib import Path
from PIL import Image

root = Path("data")

for split in ["train", "validation"]:
    print("==", split, "==")
    img_dir = root / split / "images"
    for cls_dir in sorted(img_dir.iterdir()):
        if cls_dir.is_dir():
            files = [f for f in cls_dir.iterdir() if f.is_file()]
            exts = {f.suffix.lower() for f in files}
            print(cls_dir.name, len(files), exts)

first = next((root / "train" / "images" / "crazing").iterdir())
img = Image.open(first)
print("Ornek:", first.name, img.size, img.mode)