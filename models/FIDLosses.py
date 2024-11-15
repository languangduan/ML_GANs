import torch
import torch.nn.functional as F
import numpy as np


class MiFIDLoss(torch.nn.Module):
    def __init__(self, feature_extractor, cosine_distance_eps=0.1, fid_epsilon=1e-15):
        super(MiFIDLoss, self).__init__()
        self.cosine_distance_eps = cosine_distance_eps
        self.fid_epsilon = fid_epsilon
        self.feature_extractor = feature_extractor  # 预训练模型用于提取特征
        self.mu1, self.sigma1, self.mu2, self.sigma2 = None, None, None, None

    def calculate_statistics(self, features):
        """计算给定特征的均值和协方差矩阵"""
        if torch.isnan(features).any():
            print("Features contain NaN values.")
        if torch.isinf(features).any():
            print("Features contain infinite values.")

        mu = torch.mean(features, dim=0)

        # 确保 features 是二维
        features_np = features.detach().cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np[np.newaxis, :]  # 转为二维

        if np.all(features_np == 0, axis=1).any():
            print("Input data contains zero vectors.")

        sigma = torch.from_numpy(np.cov(features_np, rowvar=False)).float()

        # 检查 sigma 是否有效
        if torch.isnan(sigma).any() or torch.isinf(sigma).any():
            raise ValueError("Covariance matrix is NaN or infinite.")

        return mu, sigma

    def extract_features(self, images):
        """使用预训练模型提取特征"""
        with torch.no_grad():
            features = self.feature_extractor(images)
        return features

    def compute_real_statistics(self, real_images_loader):
        """在真实数据集上计算 mu1 和 sigma1"""
        all_features = []
        for images in real_images_loader:
            images = images[0].to(next(self.feature_extractor.parameters()).device)
            features = self.extract_features(images)
            # 获取 features 的主输出
            if isinstance(features, tuple) or isinstance(features, torch.InceptionOutputs):
                features = features[0]  # 取主输出
            all_features.append(features)

        all_features = torch.cat(all_features, dim=0)
        self.mu1, self.sigma1 = self.calculate_statistics(all_features)

    def compute_generated_statistics(self, generated_images_loader):
        """在生成数据集上计算 mu2 和 sigma2"""
        all_features = []
        for images in generated_images_loader:
            images = images[0].to(next(self.feature_extractor.parameters()).device)
            features = self.extract_features(images)
            # 获取 features 的主输出
            if isinstance(features, tuple) or isinstance(features, torch.InceptionOutputs):
                features = features[0]  # 取主输出
            all_features.append(features)

        all_features = torch.cat(all_features, dim=0)
        self.mu2, self.sigma2 = self.calculate_statistics(all_features)

    def calculate_frechet_distance(self):
        """计算 Fréchet 距离"""
        diff = self.mu1 - self.mu2

        # 添加数值稳定性检查
        covmean = torch.sqrt(
            self.sigma1 @ self.sigma2 + self.fid_epsilon * torch.eye(self.sigma1.shape[0]).to(self.sigma1.device)
        )

        # 检查 covmean 是否有效
        if torch.isnan(covmean).any() or torch.isinf(covmean).any():
            raise ValueError("Covariance mean is NaN or infinite.")

        tr_covmean = torch.trace(covmean)
        fid_value = diff.dot(diff) + torch.trace(self.sigma1) + torch.trace(self.sigma2) - 2 * tr_covmean

        if torch.isnan(fid_value) or torch.isinf(fid_value):
            raise ValueError("FID value is NaN or infinite.")

        return fid_value
    def cosine_distance(self, features1, features2):
        # 如果输入是一维，直接归一化并计算
        if features1.dim() == 1 and features2.dim() == 1:
            if torch.norm(features1) == 0 or torch.norm(features2) == 0:
                return torch.tensor(1.0, device=features1.device)  # 设定一个默认值，避免NaN
            norm_f1 = F.normalize(features1, p=2, dim=0)
            norm_f2 = F.normalize(features2, p=2, dim=0)
            d = 1.0 - torch.dot(norm_f1, norm_f2).abs()
        else:
            # 对二维情况进行归一化，移除全零向量
            features1_nozero = features1[torch.norm(features1, dim=1) != 0]
            features2_nozero = features2[torch.norm(features2, dim=1) != 0]

            if features1_nozero.size(0) == 0 or features2_nozero.size(0) == 0:
                return torch.tensor(1.0, device=features1.device)  # 如果移除后为空，返回默认距离值

            norm_f1 = F.normalize(features1_nozero, p=2, dim=1)
            norm_f2 = F.normalize(features2_nozero, p=2, dim=1)
            d = 1.0 - torch.mm(norm_f1, norm_f2.T).abs()
            d = torch.mean(torch.min(d, dim=1)[0])

        return d

    def distance_thresholding(self, d):
        return d if d < self.cosine_distance_eps else 1.0

    def forward(self, real_images_loader, generated_images_loader):
        """计算 MiFID 损失"""
        # Compute statistics if not already computed
        if self.mu1 is None or self.sigma1 is None:
            self.compute_real_statistics(real_images_loader)
        if self.mu2 is None or self.sigma2 is None:
            self.compute_generated_statistics(generated_images_loader)

        fid_value = self.calculate_frechet_distance()
        cosine_dist = self.cosine_distance(self.mu1, self.mu2)
        cosine_dist = self.distance_thresholding(cosine_dist)

        mifid_score = fid_value / (cosine_dist + self.fid_epsilon)
        return mifid_score
