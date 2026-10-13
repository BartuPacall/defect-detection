from pathlib import Path
import torch
import torch.nn as nn
from dataset import get_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Cihaz:", device)


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1),
        nn.BatchNorm2d(o),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


train_loader, val_loader, classes = get_loaders(batch_size=32)
print("Siniflar:", classes)

model = SmallCNN(len(classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

Path("models").mkdir(exist_ok=True)
best_acc = 0.0

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
        torch.save(model.state_dict(), "models/cnn_baseline.pt")

print("En iyi val acc:", round(best_acc, 4))