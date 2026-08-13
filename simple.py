# simple.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import timm
from sklearn.metrics import f1_score, classification_report

# ======================
# 单通道数据集
# ======================
class SingleViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            class_dir = os.path.join(root_dir, cls)
            for file in os.listdir(class_dir):
                if file.endswith('_byte.png'):
                    path = os.path.join(class_dir, file)
                    self.samples.append((path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert('L')
        img = self.transform(img)
        return img, y


# ======================
# 模型
# ======================
class SingleViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)


# ======================
# 训练
# ======================
def train(model, loader, optimizer, criterion, device):
    model.train()
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
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

    dataset = SingleViewDataset(data_root, transform)
    train_set, test_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    model = SingleViT(len(dataset.class_to_idx)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        train(model, train_loader, optimizer, criterion, device)

    evaluate(model, test_loader, device)