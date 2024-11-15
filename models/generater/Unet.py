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

# 定义Upsample模块
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, size, apply_dropout=False):
        super(Upsample, self).__init__()
        # 初始化转置卷积层，使用正态分布初始化权重
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=size, stride=2, padding=1, bias=False)
        nn.init.normal_(self.conv_transpose.weight, mean=0.0, std=0.02)

        # 使用GroupNorm代替InstanceNorm
        self.norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=True)
        nn.init.normal_(self.norm.weight, mean=0.0, std=0.02)

        self.apply_dropout = apply_dropout
        if apply_dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        if self.apply_dropout:
            x = self.dropout(x)
        x = self.relu(x)
        return x

# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, output_channels=3):
        super(Generator, self).__init__()

        # Downsampling layers
        self.down1 = Downsample(3, 64, 4, apply_instancenorm=False)  # (bs, 64, 128, 128)
        self.down2 = Downsample(64, 128, 4)  # (bs, 128, 64, 64)
        self.down3 = Downsample(128, 256, 4)  # (bs, 256, 32, 32)
        self.down4 = Downsample(256, 512, 4)  # (bs, 512, 16, 16)
        self.down5 = Downsample(512, 512, 4)  # (bs, 512, 8, 8)
        self.down6 = Downsample(512, 512, 4)  # (bs, 512, 4, 4)
        self.down7 = Downsample(512, 512, 4)  # (bs, 512, 2, 2)
        self.down8 = Downsample(512, 512, 4)  # (bs, 512, 1, 1)

        # Upsampling layers
        self.up1 = Upsample(512, 512, 4, apply_dropout=True)  # (bs, 512, 2, 2)
        self.up2 = Upsample(1024, 512, 4, apply_dropout=True)  # (bs, 512, 4, 4)
        self.up3 = Upsample(1024, 512, 4, apply_dropout=True)  # (bs, 512, 8, 8)
        self.up4 = Upsample(1024, 512, 4)  # (bs, 512, 16, 16)
        self.up5 = Upsample(1024, 256, 4)  # (bs, 256, 32, 32)
        self.up6 = Upsample(512, 128, 4)  # (bs, 128, 64, 64)
        self.up7 = Upsample(256, 64, 4)  # (bs, 64, 128, 128)

        # 最后一层
        self.last = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(self.last.weight, mean=0.0, std=0.02)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)  # (bs, 64, 128, 128)
        d2 = self.down2(d1)  # (bs, 128, 64, 64)
        d3 = self.down3(d2)  # (bs, 256, 32, 32)
        d4 = self.down4(d3)  # (bs, 512, 16, 16)
        d5 = self.down5(d4)  # (bs, 512, 8, 8)
        d6 = self.down6(d5)  # (bs, 512, 4, 4)
        d7 = self.down7(d6)  # (bs, 512, 2, 2)
        d8 = self.down8(d7)  # (bs, 512, 1, 1)

        # Upsampling with skip connections
        x = self.up1(d8)  # (bs, 512, 2, 2)
        x = torch.cat([x, d7], dim=1)  # (bs, 1024, 2, 2)

        x = self.up2(x)  # (bs, 512, 4, 4)
        x = torch.cat([x, d6], dim=1)  # (bs, 1024, 4, 4)

        x = self.up3(x)  # (bs, 512, 8, 8)
        x = torch.cat([x, d5], dim=1)  # (bs, 1024, 8, 8)

        x = self.up4(x)  # (bs, 512, 16, 16)
        x = torch.cat([x, d4], dim=1)  # (bs, 1024, 16, 16)

        x = self.up5(x)  # (bs, 256, 32, 32)
        x = torch.cat([x, d3], dim=1)  # (bs, 512, 32, 32)

        x = self.up6(x)  # (bs, 128, 64, 64)
        x = torch.cat([x, d2], dim=1)  # (bs, 256, 64, 64)

        x = self.up7(x)  # (bs, 64, 128, 128)
        x = torch.cat([x, d1], dim=1)  # (bs, 128, 128, 128)

        x = self.last(x)  # (bs, 3, 256, 256)
        x = self.tanh(x)

        return x

if __name__ == '__main__':
    # 创建生成器模型
    generator = Generator()

    # 将模型移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)

    # 生成三通道的随机图像数据
    random_input = torch.randn(16, 3, 256, 256).to(device)  # batch size = 16, channels = 3, image size = 256x256

    # 通过生成器获取输出
    output = generator(random_input)
    print(output.shape)  # 应该输出 torch.Size([16, 3, 256, 256])
