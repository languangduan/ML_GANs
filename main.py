import logging
import os
import torch
from backbones_unet.model.unet import Unet
from torch import optim

from models.CycleGan import CycleGan
from models.generater.Unet import Generator
from models.generater.discriminer import Discriminator
from models.losses import gen_loss_fn, cycle_loss_fn, identity_loss_fn, disc_loss_fn
# from models.patchGAN.disc_improve import Discriminator
from models.MiFID_evaluate import MIFIDEvaluator
# from test import num_images
from utils.data_utils import create_dataloaders, save_generated_images, load_latest_checkpoint
from utils.setup_utils import parse_args, set_random_seed, setup_logger, log_hyperparameters, select_device


def train_cyclegan(cyclegan, train_loader, val_loader, num_epochs, device, model_save_dir, img_save_dir, evaluator, load=False):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if load:
        start_epoch = load_latest_checkpoint(cyclegan, model_save_dir)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        cyclegan.train()

        for i, (real_monet, real_photo) in enumerate(train_loader):
            real_monet = real_monet.to(device)
            real_photo = real_photo.to(device)

            # print(real_monet.shape)
            # 执行一个训练步骤
            losses = cyclegan.train_step(real_monet, real_photo)

            # 打印损失
            if i % 10 == 0:
                logging.info(f"Batch {i}, Monet Gen Loss: {losses['monet_gen_loss']:.4f}, "
                             f"Photo Gen Loss: {losses['photo_gen_loss']:.4f}, "
                             f"Monet Disc Loss: {losses['monet_disc_loss']:.4f}, "
                             f"Photo Disc Loss: {losses['photo_disc_loss']:.4f}")



        # 验证模型
        validate(cyclegan, val_loader, device, evaluator, epoch, img_save_dir)

        if epoch % 10 == 0 or epoch == num_epochs-1:
            # 保存模型
            model_path = os.path.join(model_save_dir, f'cyclegan_epoch_{epoch + 1}.pth')
            torch.save(cyclegan.state_dict(), model_path)
            logging.info(f'Model saved at {model_path}')


def validate(cyclegan, val_loader, device, evaluator, epoch, save_dir):
    cyclegan.eval()
    real_images = []
    generated_images = []
    with torch.no_grad():
        for real_monet, real_photo in val_loader:
            real_monet = real_monet.to(device)
            real_photo = real_photo.to(device)

            # 使用生成器生成 Monet 风格的图像
            fake_monet = cyclegan.m_gen(real_photo)

            # 添加到列表中
            real_images.append(real_monet.cpu())
            generated_images.append(fake_monet.cpu())

    # 拼接所有的图像
    real_images = torch.cat(real_images, dim=0)
    generated_images = torch.cat(generated_images, dim=0)

    # 确保图像值在 [0, 1]
    real_images = (real_images + 1) / 2
    generated_images = (generated_images + 1) / 2

    if (epoch + 1) % 5 == 0:
        # 保存生成的图像
        save_generated_images(generated_images, epoch, save_dir)

    # 使用 MIFIDEvaluator 计算 MIFID 分数
    fid_value, distance = evaluator.calculate_metrics_from_arrays(generated_images, real_images)
    fid_epsilon = 1e-15
    multiplied_score = fid_value / (distance + fid_epsilon)

    logging.info(f"Validation MIFID Score: {multiplied_score:.4f}")


def main():
    args = parse_args()
    set_random_seed(args.seed)  # 设置随机种子
    log_file = setup_logger(args.log_save_dir)  # 设置 logger
    log_hyperparameters(args, log_file)  # 记录超参数

    device = select_device(args.device)

    monet_generator = Generator().to(device)  # Unet(backbone='vgg11', in_channels=3, num_classes=3).to(device) # 'vgg11' convnext_base',
    photo_generator = Generator().to(device)  # Unet(backbone='vgg11', in_channels=3, num_classes=3).to(device)
    monet_discriminator = Discriminator().to(device) # Discriminator(input_nc=3).to(device)
    photo_discriminator = Discriminator().to(device) # Discriminator(input_nc=3).to(device)

    cyclegan = CycleGan(
        monet_generator=monet_generator,
        photo_generator=photo_generator,
        monet_discriminator=monet_discriminator,
        photo_discriminator=photo_discriminator
    )
    betas = (args.beta1, args.beta2)
    m_gen_optimizer = optim.Adam(monet_generator.parameters(), lr=args.lr, betas=betas)
    p_gen_optimizer = optim.Adam(photo_generator.parameters(), lr=args.lr, betas=betas)
    m_disc_optimizer = optim.Adam(monet_discriminator.parameters(), lr=args.lr, betas=betas)
    p_disc_optimizer = optim.Adam(photo_discriminator.parameters(), lr=args.lr, betas=betas)

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
    # 初始化 MIFIDEvaluator
    evaluator = MIFIDEvaluator(device)

    # 训练循环
    train_loader, val_loader = create_dataloaders(args.data_dir, args.batch_size)
    train_cyclegan(cyclegan, train_loader, val_loader, args.epochs, device, args.model_save_dir,
                   args.img_save_dir, evaluator)


if __name__ == '__main__':
    main()
