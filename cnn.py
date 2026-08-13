# cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import f1_score, classification_report

# ======================
# 数据集
# ======================
class ThreeViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            class_dir = os.path.join(root_dir, cls)
            for file in os.listdir(class_dir):
                if file.endswith('_byte.png'):
                    base = file.replace('_byte.png', '')
                    byte_path = os.path.join(class_dir, base + '_byte.png')
                    asm_path = os.path.join(class_dir, base + '_asm.png')
                    api_path = os.path.join(class_dir, base + '_api.png')

                    if os.path.exists(asm_path) and os.path.exists(api_path):
                        self.samples.append((byte_path, asm_path, api_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b, a, p, y = self.samples[idx]
        b = self.transform(Image.open(b).convert('L'))
        a = self.transform(Image.open(a).convert('L'))
        p = self.transform(Image.open(p).convert('L'))
        return b, a, p, y


# ======================
# CNN模型
# ======================
class ThreeViewCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def branch():
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )

        self.b1 = branch()
        self.b2 = branch()
        self.b3 = branch()

        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x1, x2, x3):
        f1 = self.b1(x1).view(x1.size(0), -1)
        f2 = self.b2(x2).view(x2.size(0), -1)
        f3 = self.b3(x3).view(x3.size(0), -1)
        return self.fc(torch.cat([f1, f2, f3], dim=1))


# ======================
# 训练 + 测试
# ======================
def train(model, loader, optimizer, criterion, device):
    model.train()
    for b,a,p,y in loader:
        b,a,p,y = b.to(device),a.to(device),p.to(device),y.to(device)
        optimizer.zero_grad()
        out = model(b,a,p)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for b,a,p,y in loader:
            b,a,p = b.to(device),a.to(device),p.to(device)
            out = model(b,a,p)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    f1 = f1_score(labels, preds, average='weighted')
    print("F1:", f1)
    print(classification_report(labels, preds, zero_division=0))


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    data_root = "./dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])

    dataset = ThreeViewDataset(data_root, transform)
    train_set, test_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    model = ThreeViewCNN(len(dataset.class_to_idx)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        train(model, train_loader, optimizer, criterion, device)

    evaluate(model, test_loader, device)