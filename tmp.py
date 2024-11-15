import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np




def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(299),  # Inception v3需要299x299的输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # 增加一个批次维度
    return image

def extract_features(image_tensor):
    with torch.no_grad():  # 不需要计算梯度
        features = model(image_tensor)
    return features


# 生成随机图像
def generate_random_image(size=(299, 299, 3)):
    return np.random.rand(*size) * 255  # 生成随机RGB图像


# 生成多个随机图像并提取特征
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 加载预训练的Inception v3模型
model = models.inception_v3(pretrained=True)

# 去除最后一层（分类层）
# 这里我们只保留特征提取部分
model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
model.eval()  # 设置为评估模式

num_images = 5
for i in range(num_images):
    # random_image = generate_random_image()
    # random_image_tensor = preprocess_image(random_image).to(device)
    real_images = torch.rand(16, 3, 299, 299).to(device)

    # 提取特征
    features = extract_features(real_images)

    # 输出特征的形状
    print(f"Extracted features shape for image {i + 1}:", features.shape)
