import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义Downsample模块
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, size, apply_instancenorm=True):
        super(Downsample, self).__init__()
        # 初始化卷积层，使用正态分布初始化权重
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=2, padding=1, bias=False)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)

        # 是否应用GroupNorm
        self.apply_instancenorm = apply_instancenorm
        if apply_instancenorm:
            # 使用GroupNorm代替InstanceNorm，groups设置为通道数，实现InstanceNorm的效果
            self.norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=True)
            nn.init.normal_(self.norm.weight, mean=0.0, std=0.02)
        else:
            self.norm = None

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.apply_instancenorm:
            x = self.norm(x)
        x = self.leaky_relu(x)
        return x


# 实现判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Downsample layers
        # 注意这里正确设置了输入通道数
        self.down1 = Downsample(in_channels=3, out_channels=64, size=4, apply_instancenorm=False)  # (bs, 64, 128, 128)
        self.down2 = Downsample(in_channels=64, out_channels=128, size=4, apply_instancenorm=True)  # (bs, 128, 64, 64)
        self.down3 = Downsample(in_channels=128, out_channels=256, size=4, apply_instancenorm=True)  # (bs, 256, 32, 32)

        # 中间卷积层
        self.zero_pad1 = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(256, 512, kernel_size=4, stride=1, bias=False)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)

        # 使用GroupNorm，groups=512实现Instance Normalization的效果
        self.norm1 = nn.GroupNorm(num_groups=512, num_channels=512, affine=True)
        nn.init.normal_(self.norm1.weight, mean=0.0, std=0.02)

        self.leaky_relu = nn.LeakyReLU(0.2)

        # 最后的卷积层
        self.zero_pad2 = nn.ZeroPad2d(1)
        self.last = nn.Conv2d(512, 1, kernel_size=4, stride=1)
        self.sigmoid = nn.Sigmoid()
        nn.init.normal_(self.last.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x的形状应该是(batch_size, 3, 256, 256)

        x = self.down1(x)  # (bs, 64, 128, 128)
        x = self.down2(x)  # (bs, 128, 64, 64)
        x = self.down3(x)  # (bs, 256, 32, 32)

        x = self.zero_pad1(x)  # (bs, 256, 34, 34)
        x = self.conv(x)  # (bs, 512, 31, 31)
        x = self.norm1(x)
        x = self.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 512, 33, 33)
        x = self.last(x)  # (bs, 1, 30, 30)
        x = self.sigmoid(x)
        return x


# 测试代码
def test_discriminator():
    # 创建判别器实例
    discriminator = Discriminator()

    # 将模型移到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = discriminator.to(device)

    # 生成随机的测试数据
    batch_size = 16
    test_input = torch.randn(batch_size, 3, 256, 256).to(device)

    # 运行模型
    with torch.no_grad():
        output = discriminator(test_input)

    # 打印输出形状
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # 检查参数数量
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Total number of parameters: {total_params:,}")

    return discriminator


# 运行测试
if __name__ == "__main__":
    discriminator = test_discriminator()
