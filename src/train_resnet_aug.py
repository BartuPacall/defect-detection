import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from dataset_aug import get_loaders_aug

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Cihaz:", device, "| seed:", seed, "| artirma: acik")

train_loader, val_loader, classes = get_loaders_aug(batch_size=32)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

Path("models").mkdir(exist_ok=True)
best_acc, val_acc = 0.0, 0.0

for epoch in range(1, 21):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(x)
    train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    val_acc = correct / len(val_loader.dataset)

    print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | val acc {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"models/resnet18_aug_seed{seed}.pt")

print("En iyi val acc:", round(best_acc, 4))
print("Son epoch val acc:", round(val_acc, 4))