import argparse
import os
import random

import numpy as np
import torch
import logging
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train CycleGAN model")
    parser.add_argument('--data_dir', type=str, default='datasets', help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for optimizers')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for optimizers')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument("--img_save_dir", type=str, default="generated_imgs", help="Directory to save generated images")
    parser.add_argument("--log_save_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Weight for cycle consistency loss')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(save_dir):
    # 获取当前时间并格式化为字符串，用于文件名
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(save_dir, f'training_{current_time}.log')
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 创建屏幕处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return log_file


def log_hyperparameters(args, log_file):
    logging.info("Hyperparameters:")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")


def select_device(preferred_device):
    # 如果指定设备可用，使用它
    if torch.cuda.is_available() and preferred_device.startswith('cuda'):
        device_id = int(preferred_device.split(':')[1])  # 获取设备 ID
        if device_id < torch.cuda.device_count():
            print(f"Using device: {preferred_device}")
            return torch.device(preferred_device)
        else:
            print(f"Specified device {preferred_device} not available, selecting alternate device.")

    # 如果指定设备不可用，按优先级选择 cuda:0 或 cpu
    if torch.cuda.is_available():
        print("Using device: cuda:0")
        return torch.device("cuda:0")
    else:
        print("Using device: cpu")
        return torch.device("cpu")
