import torch
import torch.nn as nn
import torch.optim as optim

from models.CycleGan import CycleGan
from backbones_unet.model.unet import Unet
from models.patchGAN.disc_improve import Discriminator

# 启用异常检测
torch.autograd.set_detect_anomaly(True)


# 假设这是一个简单的生成器类
class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),  # 修改为inplace=False
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化生成器和判别器
monet_generator = Unet(
        backbone='convnext_base',  # backbone network name
        in_channels=3,  # input channels (1 for gray-scale images, 3 for RGB, etc.)
        num_classes=3,  # output channels (number of classes in your dataset)
    ).to(device)
photo_generator = Unet(
        backbone='convnext_base',  # backbone network name
        in_channels=3,  # input channels (1 for gray-scale images, 3 for RGB, etc.)
        num_classes=3,  # output channels (number of classes in your dataset)
    ).to(device)
monet_discriminator = Discriminator(input_nc=3).to(device)
photo_discriminator = Discriminator(input_nc=3).to(device)

# 初始化CycleGAN模型
cyclegan = CycleGan(
    monet_generator=monet_generator,
    photo_generator=photo_generator,
    monet_discriminator=monet_discriminator,
    photo_discriminator=photo_discriminator
)

# 定义优化器和损失函数
m_gen_optimizer = optim.Adam(monet_generator.parameters(), lr=0.0002)
p_gen_optimizer = optim.Adam(photo_generator.parameters(), lr=0.0002)
m_disc_optimizer = optim.Adam(monet_discriminator.parameters(), lr=0.0002)
p_disc_optimizer = optim.Adam(photo_discriminator.parameters(), lr=0.0002)

def gen_loss_fn(fake_output):
    return nn.BCELoss()(fake_output, torch.ones_like(fake_output))

def disc_loss_fn(real_output, fake_output):
    real_loss = nn.BCELoss()(real_output, torch.ones_like(real_output))
    fake_loss = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
    return (real_loss + fake_loss) / 2

def cycle_loss_fn(real, cycled, lambda_cycle):
    return lambda_cycle * nn.L1Loss()(real, cycled)

def identity_loss_fn(real, same, lambda_cycle):
    return lambda_cycle * nn.L1Loss()(real, same)

# 编译CycleGAN
cyclegan.compile(
    m_gen_optimizer=m_gen_optimizer,
    p_gen_optimizer=p_gen_optimizer,
    m_disc_optimizer=m_disc_optimizer,
    p_disc_optimizer=p_disc_optimizer,
    gen_loss_fn=gen_loss_fn,
    disc_loss_fn=disc_loss_fn,
    cycle_loss_fn=cycle_loss_fn,
    identity_loss_fn=identity_loss_fn
)

# 创建一些假数据进行测试
real_monet = torch.randn(4, 3, 256, 256).to(device)
real_photo = torch.randn(4, 3, 256, 256).to(device)

# 运行一个训练步骤
losses = cyclegan.train_step(real_monet, real_photo)

# 打印损失
print("Monet Generator Loss:", losses["monet_gen_loss"])
print("Photo Generator Loss:", losses["photo_gen_loss"])
print("Monet Discriminator Loss:", losses["monet_disc_loss"])
print("Photo Discriminator Loss:", losses["photo_disc_loss"])
