import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dataset import get_loaders
from train_cnn_model import SmallCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
_, val_loader, classes = get_loaders()

model = SmallCNN(len(classes)).to(device)
model.load_state_dict(torch.load("models/cnn_baseline.pt", map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in val_loader:
        pred = model(x.to(device)).argmax(1).cpu()
        y_true += y.tolist()
        y_pred += pred.tolist()

print(classification_report(y_true, y_pred, target_names=classes, digits=3))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout()
Path("reports").mkdir(exist_ok=True)
plt.savefig("reports/cm_cnn_baseline.png", dpi=120)
print("Kaydedildi: reports/cm_cnn_baseline.png")