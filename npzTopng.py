import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# 1. 读取数据
data = np.load('malimg.npz', allow_pickle=True)
arr = data['arr']

X, y = [], []

for sample in arr:
    img, label = sample
    X.append(img)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Loaded:", X.shape, y.shape)

# 2. 输出目录
output_dir = './dataset'
os.makedirs(output_dir, exist_ok=True)

# 3. 归一化
def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

# 4. 三通道构造
def create_three_views(img):
    img = normalize(img)

    # Byte
    byte_img = img

    # ASM（旋转模拟结构）
    asm_img = np.rot90(img)

    # API（噪声模拟行为）
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    api_img = np.clip(img + noise, 0, 255)

    return byte_img, asm_img, api_img

# 5. 生成PNG
for i in tqdm(range(len(X))):
    img = X[i]
    label = str(y[i])

    class_dir = os.path.join(output_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    byte_img, asm_img, api_img = create_three_views(img)

    base = os.path.join(class_dir, f"{i}")

    Image.fromarray(byte_img).resize((64,64)).save(base + '_byte.png')
    Image.fromarray(asm_img).resize((64,64)).save(base + '_asm.png')
    Image.fromarray(api_img).resize((64,64)).save(base + '_api.png')

print("✅ 数据集生成完成！")