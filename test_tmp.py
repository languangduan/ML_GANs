import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import numpy as np

from models.FIDLosses import MiFIDLoss


# 定义简单的虚假数据生成函数
def generate_fake_data(num_samples, num_channels, height, width):
    return torch.randn(num_samples, num_channels, height, width)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 定义测试特征提取模型（使用预训练模型）
feature_extractor = models.inception_v3(pretrained=True).to(device)
feature_extractor.fc = torch.nn.Identity()  # 移除最后一层全连接层，直接获取特征

# 生成虚假数据
num_samples = 100
real_images = generate_fake_data(num_samples, 3, 299, 299).to(device)  # Inception 预期的输入尺寸是 299x299
generated_images = generate_fake_data(num_samples, 3, 299, 299).to(device)

# 创建数据加载器
batch_size = 10
real_images_loader = DataLoader(TensorDataset(real_images), batch_size=batch_size)
generated_images_loader = DataLoader(TensorDataset(generated_images), batch_size=batch_size)

# 初始化 MiFID 损失函数
mifid_loss_fn = MiFIDLoss(feature_extractor)

# 计算 MiFID 损失
mifid_loss = mifid_loss_fn(real_images_loader, generated_images_loader)
print("MiFID Loss:", mifid_loss.item())
