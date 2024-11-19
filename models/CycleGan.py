import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import ImageBuffer


class CycleGan(nn.Module):
    def __init__(self, monet_generator, photo_generator, monet_discriminator, photo_discriminator, lambda_cycle=10, buffer_size=50):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        self.fake_monet_buffer = ImageBuffer(buffer_size)
        self.fake_photo_buffer = ImageBuffer(buffer_size)

    def compile(self, m_gen_optimizer, p_gen_optimizer, m_disc_optimizer, p_disc_optimizer, gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn):
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, real_monet, real_photo):
        # 清零所有优化器的梯度
        self.m_gen_optimizer.zero_grad()
        self.p_gen_optimizer.zero_grad()
        self.m_disc_optimizer.zero_grad()
        self.p_disc_optimizer.zero_grad()

        # 前向传播
        # 1. 生成图像
        fake_monet = self.m_gen(real_photo)
        cycled_photo = self.p_gen(fake_monet)

        fake_photo = self.p_gen(real_monet)
        cycled_monet = self.m_gen(fake_photo)

        # 2. 生成自身
        same_monet = self.m_gen(real_monet)
        same_photo = self.p_gen(real_photo)

        # 3. 使用buffer获取历史假图像
        fake_monet_buffer = self.fake_monet_buffer(fake_monet)
        fake_photo_buffer = self.fake_photo_buffer(fake_photo)

        # 4. 判别器输出
        disc_real_monet = self.m_disc(real_monet)
        disc_real_photo = self.p_disc(real_photo)

        # 使用buffer中的图像进行判别
        disc_fake_monet = self.m_disc(fake_monet_buffer)
        disc_fake_photo = self.p_disc(fake_photo_buffer)

        # 计算损失
        # 1. 生成器损失
        monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
        photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

        # 2. 循环一致性损失
        total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + \
                           self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

        # 3. 总的生成器损失
        total_monet_gen_loss = monet_gen_loss + \
                               self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle) + \
                               total_cycle_loss
        total_photo_gen_loss = photo_gen_loss + \
                               self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle) + \
                               total_cycle_loss

        # 4. 判别器损失
        monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet.detach())
        photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo.detach())

        # 反向传播和优化
        # 1. 生成器反向传播
        total_monet_gen_loss.backward(retain_graph=True)
        total_photo_gen_loss.backward(retain_graph=True)

        # 2. 判别器反向传播
        monet_disc_loss.backward(retain_graph=True)
        photo_disc_loss.backward()

        # 更新参数
        self.m_gen_optimizer.step()
        self.p_gen_optimizer.step()
        self.m_disc_optimizer.step()
        self.p_disc_optimizer.step()

        return {
            "monet_gen_loss": total_monet_gen_loss.item(),
            "photo_gen_loss": total_photo_gen_loss.item(),
            "monet_disc_loss": monet_disc_loss.item(),
            "photo_disc_loss": photo_disc_loss.item()
        }
