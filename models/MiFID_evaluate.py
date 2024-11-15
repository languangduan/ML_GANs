import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import torch.nn.functional as F
from scipy import linalg
import warnings


class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionFeatureExtractor, self).__init__()
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        # 去除 Inception v3 的最后一个全连接层
        self.inception.fc = nn.Identity()
        self.inception.eval()

    def forward(self, x):
        # x 的形状应为 [N, 3, 299, 299]
        with torch.no_grad():
            features = self.inception(x)
        return features


class MIFIDEvaluator:
    def __init__(self, device):
        self.device = device
        self.inception = InceptionFeatureExtractor().to(self.device)
        self.inception.eval()
        self.model_params = {
            'imsize': 299,
            'output_shape': 2048,
            'cosine_distance_eps': 0.1
        }

    def get_activations(self, images, batch_size=50):
        # images 的形状应为 [N, 3, H, W]，像素值范围为 [0, 1]
        n_images = images.shape[0]
        batch_size = min(batch_size, n_images)
        n_batches = n_images // batch_size + int(n_images % batch_size != 0)
        pred_arr = np.empty((n_images, self.model_params['output_shape']))

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_images)
            batch = images[start:end].to(self.device)
            # 调整图像大小到 299x299
            batch = F.interpolate(batch, size=(self.model_params['imsize'], self.model_params['imsize']),
                                  mode='bilinear', align_corners=False)
            # Inception v3 预处理
            batch = batch * 2 - 1  # 将 [0, 1] 映射到 [-1, 1]
            pred = self.inception(batch).cpu().numpy()
            pred_arr[start:end] = pred.reshape(-1, self.model_params['output_shape'])
        return pred_arr

    def calculate_activation_statistics(self, images, batch_size=50):
        act = self.get_activations(images, batch_size)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma, act

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "训练和测试均值向量长度不一致"
        assert sigma1.shape == sigma2.shape, "训练和测试协方差矩阵尺寸不一致"

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "FID 计算产生非有限的结果；在协方差矩阵对角线上添加 %s" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("协方差矩阵中存在非忽略的虚部：{}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return fid

    def cosine_distance(self, features1, features2):
        features1_nozero = features1[np.sum(features1, axis=1) != 0]
        features2_nozero = features2[np.sum(features2, axis=1) != 0]
        norm_f1 = self.normalize_rows(features1_nozero)
        norm_f2 = self.normalize_rows(features2_nozero)
        d = 1.0 - np.abs(np.matmul(norm_f1, norm_f2.T))
        mean_min_d = np.mean(np.min(d, axis=1))
        return mean_min_d

    @staticmethod
    def normalize_rows(x: np.ndarray):
        return np.nan_to_num(x / np.linalg.norm(x, ord=2, axis=1, keepdims=True))

    def distance_thresholding(self, d, eps):
        return d if d < eps else 1

    def calculate_metrics_from_arrays(self, gen_images, real_images, batch_size=50):
        # 确保图像是 [N, 3, H, W]，值在 [0, 1]
        m1, s1, features1 = self.calculate_activation_statistics(gen_images, batch_size)
        m2, s2, features2 = self.calculate_activation_statistics(real_images, batch_size)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        distance = self.cosine_distance(features1, features2)
        distance = self.distance_thresholding(distance, self.model_params['cosine_distance_eps'])
        return fid_value, distance
