import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, real, generated):
        real_loss = self.criterion(real, torch.ones_like(real))
        generated_loss = self.criterion(generated, torch.zeros_like(generated))
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, generated):
        return self.criterion(generated, torch.ones_like(generated))

class CycleLoss(nn.Module):
    def __init__(self, lambda_cycle):
        super(CycleLoss, self).__init__()
        self.lambda_cycle = lambda_cycle

    def forward(self, real_image, cycled_image):
        loss = torch.mean(torch.abs(real_image - cycled_image))
        return self.lambda_cycle * loss

class IdentityLoss(nn.Module):
    def __init__(self, lambda_cycle):
        super(IdentityLoss, self).__init__()
        self.lambda_cycle = lambda_cycle

    def forward(self, real_image, same_image):
        loss = torch.mean(torch.abs(real_image - same_image))
        return self.lambda_cycle * 0.5 * loss

def gen_loss_fn(fake_output):
    return nn.BCELoss()(fake_output, torch.ones_like(fake_output))

def disc_loss_fn(real_output, fake_output):
    real_loss = nn.BCELoss()(real_output, torch.ones_like(real_output))
    fake_loss = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
    return (real_loss + fake_loss) / 2

def cycle_loss_fn(real, cycled, lambda_cycle, lambda_idt=0.5):
    return lambda_idt * lambda_cycle * nn.L1Loss()(real, cycled)

def identity_loss_fn(real, same, lambda_cycle):
    return lambda_cycle/5 * nn.L1Loss()(real, same)
