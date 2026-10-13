import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
alias = {"pitted": "pitted_surface", "pitted_surface": "pitted_surface",
         "rolled-in_scale": "rolled-in_scale", "rolled-in": "rolled-in_scale"}

root = Path("data")
out = Path("yolo_data")

for split in ["train", "validation"]:
    split_out = "val" if split == "validation" else "train"
    (out / "images" / split_out).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split_out).mkdir(parents=True, exist_ok=True)

    n_img, n_box, missing = 0, 0, 0
    for img_path in sorted((root / split / "images").rglob("*.jpg")):
        xml_path = root / split / "annotations" / (img_path.stem + ".xml")
        if not xml_path.exists():
            missing += 1
            continue
        tree = ET.parse(xml_path).getroot()
        w = float(tree.find("size/width").text)
        h = float(tree.find("size/height").text)
        lines = []
        for obj in tree.findall("object"):
            name = obj.find("name").text.strip()
            name = alias.get(name, name)
            if name not in classes:
                print("Bilinmeyen sinif:", name, xml_path.name)
                continue
            b = obj.find("bndbox")
            x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
            x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{classes.index(name)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        shutil.copy(img_path, out / "images" / split_out / img_path.name)
        (out / "labels" / split_out / (img_path.stem + ".txt")).write_text("\n".join(lines))
        n_img += 1
        n_box += len(lines)
    print(f"{split}: {n_img} gorsel, {n_box} kutu, etiketi olmayan: {missing}")

yaml = (
    f"path: {out.resolve().as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes))
)
Path("neu_det.yaml").write_text(yaml)
print("Yazildi: neu_det.yaml")