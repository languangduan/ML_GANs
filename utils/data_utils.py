import logging
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from torchvision.transforms import transforms
from torchvision.utils import save_image


class MonetImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")  # 转换为RGB格式
        if self.transform:
            image = self.transform(image)
        return image


class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx):
        item1 = self.dataset1[idx]
        item2 = self.dataset2[idx]
        return item1, item2



def create_dataloaders(data_dir, batch_size):
    img_height = 256
    img_width = 256

    # 定义数据增强和预处理的转换
    monet_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomCrop((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2)
    ])

    photo_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2)
    ])

    # 加载数据集
    monet_dataset = MonetImageDataset(os.path.join(data_dir, 'monet_jpg'), transform=monet_transform)
    photo_dataset = MonetImageDataset(os.path.join(data_dir, 'photo_jpg'), transform=photo_transform)

    # 随机抽取300张图像用于验证
    monet_val_indices = random.sample(range(len(monet_dataset)), 300)
    photo_val_indices = random.sample(range(len(photo_dataset)), 300)

    # 创建验证数据集
    monet_val_dataset = Subset(monet_dataset, monet_val_indices)
    photo_val_dataset = Subset(photo_dataset, photo_val_indices)

    # 创建训练数据集，排除验证集中的图像
    # monet_train_indices = list(set(range(len(monet_dataset))) - set(monet_val_indices))
    photo_train_indices = list(set(range(len(photo_dataset))) - set(photo_val_indices))

    monet_train_dataset = monet_dataset # Subset(monet_dataset, monet_train_indices)
    photo_train_dataset = Subset(photo_dataset, photo_train_indices)

    # 创建自定义数据集
    train_dataset = PairedDataset(monet_train_dataset, photo_train_dataset)
    val_dataset = PairedDataset(monet_val_dataset, photo_val_dataset)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logging.info(
        f"Generated dataloaders from {data_dir} with {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
    )

    return train_loader, val_loader


def create_dataloaders_old(data_dir, batch_size, val_split=0.1):
    img_height = 256
    img_width = 256
    # 定义数据增强和预处理的转换
    monet_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),  # 调整图像大小
        transforms.RandomCrop((img_height, img_width)),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为 Tensor，并归一化到 [0, 1]
        transforms.Lambda(lambda x: (x - 0.5) * 2)  # 归一化到 [-1, 1]
    ])

    photo_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),  # 调整图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为 Tensor，并归一化到 [0, 1]
        transforms.Lambda(lambda x: (x - 0.5) * 2)  # 归一化到 [-1, 1]
    ])

    # 加载数据集
    monet_dataset = MonetImageDataset(os.path.join(data_dir, 'monet_jpg'), transform=monet_transform)
    photo_dataset = MonetImageDataset(os.path.join(data_dir, 'photo_jpg'), transform=photo_transform)

    # 创建自定义数据集，将所有数据用于训练
    full_train_dataset = PairedDataset(monet_dataset, photo_dataset)

    # 随机抽取一部分数据用于验证
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    split = int(val_split * dataset_size)

    # 随机打乱索引
    random.shuffle(indices)

    # 划分训练和验证索引
    train_indices, val_indices = indices, indices[:split]

    # 创建训练和验证数据集
    train_dataset = full_train_dataset  # 使用完整数据集
    val_dataset = Subset(full_train_dataset, val_indices)  # 使用一部分数据集进行验证

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logging.info(
        f"Generated dataloaders from {data_dir} with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    # # 分割数据集
    # train_size = int(0.8 * len(monet_dataset))
    # val_size = len(monet_dataset) - train_size
    # monet_train, monet_val = random_split(monet_dataset, [train_size, val_size])
    #
    # train_size = int(0.8 * len(photo_dataset))
    # val_size = len(photo_dataset) - train_size
    # photo_train, photo_val = random_split(photo_dataset, [train_size, val_size])
    #
    # # 创建自定义数据集
    # train_dataset = PairedDataset(monet_train, photo_train)
    # val_dataset = PairedDataset(monet_val, photo_val)
    #
    # # 创建 DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logging.info(f"Generated dataloaders from {data_dir}")
    return train_loader, val_loader


def save_generated_images(images, epoch, save_dir):
    save_path = os.path.join(save_dir, 'generated_images', f'epoch_{epoch}')
    os.makedirs(save_path, exist_ok=True)
    for idx, image in enumerate(images):
        save_image(image, os.path.join(save_path, f"generated_{idx}.png"))
    logging.info(f"Generated images saved at {save_path}")


def load_latest_checkpoint(model, model_save_dir):
    """加载最新的模型检查点"""
    if not os.path.exists(model_save_dir):
        return 0  # 如果目录不存在，返回初始epoch为0

    # 获取目录下所有的.pth文件
    checkpoints = [f for f in os.listdir(model_save_dir) if f.endswith('.pth')]
    if not checkpoints:
        return 0  # 如果没有检查点文件，返回初始epoch为0

    # 找到最新的检查点
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])

    # 加载模型权重
    checkpoint_path = os.path.join(model_save_dir, latest_checkpoint)
    model.load_state_dict(torch.load(checkpoint_path))
    logging.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {latest_epoch})")

    return latest_epoch


class ImageBuffer:
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.buffer = []
        self.num_imgs = 0

    def __call__(self, images):
        if self.buffer_size == 0:
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)  # 添加batch维度

            if self.num_imgs < self.buffer_size:
                self.buffer.append(image)
                self.num_imgs += 1
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    random_idx = random.randint(0, self.buffer_size - 1)
                    temp = self.buffer[random_idx].clone()
                    self.buffer[random_idx] = image
                    return_images.append(temp)
                else:
                    return_images.append(image)

        return torch.cat(return_images, 0)