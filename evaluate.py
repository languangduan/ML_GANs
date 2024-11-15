import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from scipy.linalg import sqrtm

# 使用预训练的 Inception 网络
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionFeatureExtractor, self).__init__()
        inception = models.inception_v3(pretrained=True)
        self.feature_extractor = inception # nn.Sequential(*list(inception.children())[:-1])

    def forward(self, x):
        return self.feature_extractor(x)

# 计算特征均值和协方差
def calculate_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

# 计算 FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# 计算 MiFID
def calculate_mifid(fid, distances, epsilon=0.1):
    d_thr = np.mean([min(d, 1.0) for d in distances if d < epsilon])
    mifid = fid / d_thr
    return mifid

# 示例数据（假设已提取特征）
real_features = np.random.rand(100, 2048)
generated_features = np.random.rand(100, 2048)

# 计算均值和协方差
mu_real, sigma_real = calculate_statistics(real_features)
mu_generated, sigma_generated = calculate_statistics(generated_features)

# 计算 FID
fid = calculate_fid(mu_real, sigma_real, mu_generated, sigma_generated)

# 计算记忆距离（假设计算出的一些距离）
distances = np.random.rand(100)

# 计算 MiFID
mifid = calculate_mifid(fid, distances)

print(f"FID: {fid}, MiFID: {mifid}")
