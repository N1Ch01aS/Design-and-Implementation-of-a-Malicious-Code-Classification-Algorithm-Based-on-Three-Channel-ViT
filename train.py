import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import timm
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch import amp

# ======================
# 1. Dataset
# ======================
class ThreeViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            class_dir = os.path.join(root_dir, cls)
            for file in os.listdir(class_dir):
                if file.endswith('_byte.png'):
                    base = file.replace('_byte.png', '')
                    b = os.path.join(class_dir, base + '_byte.png')
                    a = os.path.join(class_dir, base + '_asm.png')
                    p = os.path.join(class_dir, base + '_api.png')

                    if os.path.exists(a) and os.path.exists(p):
                        self.samples.append((b, a, p, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b, a, p, y = self.samples[idx]

        b = Image.open(b).convert('RGB')
        a = Image.open(a).convert('RGB')
        p = Image.open(p).convert('RGB')

        if self.transform:
            b = self.transform(b)
            a = self.transform(a)
            p = self.transform(p)

        return b, a, p, y


# ======================
# 2. Model
# ======================
class ThreeViewViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vit_b = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
        self.vit_a = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
        self.vit_p = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)

        dim = self.vit_b.num_features

        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, b, a, p):
        f1 = self.vit_b(b)
        f2 = self.vit_a(a)
        f3 = self.vit_p(p)

        x = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(x)


# ======================
# 3. Mixup
# ======================
def mixup(x1, x2, x3, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(y.size(0)).to(y.device)

    x1 = lam * x1 + (1 - lam) * x1[index]
    x2 = lam * x2 + (1 - lam) * x2[index]
    x3 = lam * x3 + (1 - lam) * x3[index]

    return x1, x2, x3, y, y[index], lam


# ======================
# 4. Train
# ======================
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for b, a, p, y in loader:
        b, a, p, y = b.to(device), a.to(device), p.to(device), y.to(device)

        b, a, p, y_a, y_b, lam = mixup(b, a, p, y)

        optimizer.zero_grad()

        with amp.autocast(device_type='cuda'):
            out = model(b, a, p)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(loader), 100 * correct / total


# ======================
# 5. Eval
# ======================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, labels = [], []
    loss_sum = 0

    for b, a, p, y in loader:
        b, a, p, y = b.to(device), a.to(device), p.to(device), y.to(device)

        out = model(b, a, p)
        loss = criterion(out, y)

        loss_sum += loss.item()
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    f1 = f1_score(labels, preds, average="weighted")
    return loss_sum / len(loader), f1, preds, labels


# ======================
# 6. Plot tools
# ======================
def plot_curves(train_loss, test_loss, train_acc, test_f1):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(train_acc, label="Train Acc")
    plt.plot(test_f1, label="Test F1")
    plt.legend()
    plt.title("Accuracy & F1")
    plt.savefig("acc_f1_curve.png", dpi=300)
    plt.close()


def plot_cm(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()


def plot_f1(labels, preds, num_classes):
    f1s = f1_score(labels, preds, average=None)

    plt.figure(figsize=(12, 5))
    plt.bar(range(num_classes), f1s)
    plt.title("Per-Class F1")
    plt.savefig("class_f1.png", dpi=300)
    plt.close()


def save_report(labels, preds):
    report = classification_report(labels, preds, digits=4, zero_division=0)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print(report)


# ======================
# 7. Main
# ======================
if __name__ == "__main__":

    data_root = "./dataset"
    batch_size = 32
    epochs = 80
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = ThreeViewDataset(data_root, transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # sampler（解决类别不均衡）
    labels = [dataset.samples[i][3] for i in train_set.indices]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = ThreeViewViT(num_classes=len(dataset.class_to_idx)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = amp.GradScaler()

    best_f1 = 0

    train_loss_list, test_loss_list = [], []
    train_acc_list, test_f1_list = [], []

    # ======================
    # training loop
    # ======================
    for epoch in range(epochs):

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        test_loss, test_f1, preds, labels = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_f1_list.append(test_f1)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} F1: {test_f1:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved best model")

    # ======================
    # final eval
    # ======================
    model.load_state_dict(torch.load("best_model.pth"))

    _, _, preds, labels = evaluate(model, test_loader, criterion, device)

    # ======================
    # plots + report
    # ======================
    plot_curves(train_loss_list, test_loss_list, train_acc_list, test_f1_list)
    plot_cm(labels, preds, list(dataset.class_to_idx.keys()))
    plot_f1(labels, preds, len(dataset.class_to_idx))
    save_report(labels, preds)

    print("\n🎯 Final F1:", best_f1)