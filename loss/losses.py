import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def hard_example_mining_fastreid(dist_mat, is_pos, is_neg):

    assert len(dist_mat.size()) == 2
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def easy_postive_mining(dist_mat, is_pos, is_neg):
    assert len(dist_mat.size()) == 2
    dis_pos = dist_mat * is_pos
    dist_ap_min, _ = torch.min(dis_pos + is_neg * 1e9 + 1e9 * torch.eye(len(dist_mat)).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")), dim=1)

    dis_neg = dist_mat * is_neg
    dist_an_min, _ = torch.min(dis_neg + is_pos * 1e9, dim=1)


    return dist_ap_min, dist_an_min


def semi_hard_positive_mining(dist_mat, is_pos, is_neg, margin):
    assert len(dist_mat.size()) == 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate dist_ap_max and dist_ap_min
    dis_pos = dist_mat * is_pos + 1e9 * is_neg + 1e9 * torch.eye(len(dist_mat)).to(device)

    # Calculate dist_an_min
    dis_neg = dist_mat * is_neg
    dist_an_min, _ = torch.min(dis_neg + is_pos * 1e9, dim=1)

    mask = torch.le(dis_pos, dist_an_min+margin)
    dist_ap_semi, _ = torch.max(mask*dis_pos, dim=0)

    return dist_ap_semi, dist_an_min

def easy_negative_mining(dist_mat, is_pos, is_neg, margin):
    assert len(dist_mat.size()) == 2

    # Calculate dist_ap_max and dist_ap_min
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)

    # Calculate dist_an_min
    dis_neg = dist_mat * is_neg

    mask = torch.ge(dis_neg, dist_ap-margin)
    dist_neg_easy, _ = torch.min(mask*dis_neg + 1e9 * mask, dim=0)

    return dist_ap, dist_neg_easy


def semi_hard_negtive_mining(dist_mat, is_pos, is_neg, margin):
    assert len(dist_mat.size()) == 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate dist_ap_max and dist_ap_min
    dis_pos = dist_mat * is_pos + 1e9 * is_neg + 1e9 * torch.eye(len(dist_mat)).to(device)
    # 最难的正样本
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)

    # Calculate dist_an_min
    dis_neg = dist_mat * is_neg
   # 求最难的负样本
    dist_an_min, _ = torch.min(dis_neg + is_pos * 1e9, dim=1)

    mask = torch.le(dis_pos, dist_an_min + margin)
   # 求最中等难度正样本
    dist_ap_semi, _ = torch.max(mask * dis_pos, dim=0)

    mask = torch.ge(dis_neg, dist_ap_semi-margin)
    # 求中等难度负样本
    dist_an_semi, _ = torch.min(mask * dis_neg + 1e9 * mask, dim=0)

    return dist_ap, dist_an_semi


def acknowledge_example_mining(dist_mat, is_pos, is_neg):


    assert len(dist_mat.size()) == 2
    dist_ap = []
    dist_an = []
    dis_pos = dist_mat * is_pos
    dist_ap_max, _ = torch.max(dis_pos, dim=1)
    dist_ap_min, _ = torch.min(dis_pos + is_neg * 1e9 +  1e9 *
                               torch.eye(len(dist_mat)).to(
                                   torch.device("cuda" if torch.cuda.is_available() else "cpu")) , dim=1)
    mid_dif_ap = dist_ap_max - (dist_ap_max - dist_ap_min) / 3

    dis_neg = dist_mat * is_neg
    dist_an_min, _ = torch.min(dis_neg + is_pos * 1e9, dim=1)
    for i in range(len(dist_mat)):
        mask = torch.le(dis_pos[i], mid_dif_ap[i])
        value = torch.max(mask * dis_pos[i])
        dist_ap.append(value)


    dist_ap = torch.tensor(dist_ap)
    dist_an = torch.tensor(dist_an)
    return dist_ap, dist_an_min


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def euclidean_dist_fast_reid(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class InstanceLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(InstanceLoss, self).__init__()
        self.margin = margin  # margin 可以用来控制同类样本之间的最小距离

    def forward(self, embeddings, labels):
        """
        :param embeddings: 模型输出的嵌入特征 (N, D)
        :param labels: 样本标签 (N,)
        :return: 返回的 Instance Loss
        """
        N = embeddings.size(0)
        loss = 0.0

        for i in range(N):
            for j in range(N):
                if labels[i] == labels[j]:  # 如果是同类样本
                    # 计算同类样本之间的距离，并最小化它
                    dist = F.pairwise_distance(embeddings[i:i+1], embeddings[j:j+1], p=2)
                    loss += dist

        loss = loss / (N * (N - 1))  # 计算平均损失
        return loss

class CircleLoss(torch.nn.Module):
    def __init__(self, margin=0.35, scale=64.0):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        # Calculate pairwise cosine similarity
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        # Create label mask: positive samples where labels match
        labels = labels.unsqueeze(1)
        positive_mask = labels == labels.T

        # Calculate positive and negative distances
        positive_similarity = similarity_matrix * positive_mask.float()
        negative_similarity = similarity_matrix * (1 - positive_mask.float())

        # Margin-based loss computation
        loss = torch.log(1 + torch.exp(-self.scale * (positive_similarity - negative_similarity - self.margin)))
        loss = loss.mean()
        return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.2, scale=64.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, embedding, targets):
        # 计算样本对之间的欧式距离
        dist_mat = euclidean_dist_fast_reid(embedding, embedding)

        # 获取样本数量
        N = dist_mat.size(0)

        # 判断正样本对
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        # 判断负样本对
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # 对于正样本对 (y=1), 计算 d^2
        pos_loss = is_pos * torch.pow(dist_mat, 2) * self.scale

        # 对于负样本对 (y=0), 计算 max(0, margin - d)^2
        neg_loss = is_neg * torch.pow(torch.clamp(self.margin - dist_mat, min=0.0), 2) * self.scale

        # 总损失
        loss = torch.mean(pos_loss + neg_loss) / 2.0

        return loss

class metric_loss_function(nn.Module):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""
    def __init__(self, data, margin, norm_feat, loss_function) -> None:
        super().__init__()
        self.margin = margin
        self.norm_feat=norm_feat
        self.loss_function = loss_function
        self.circle_loss = CircleLoss(data['margin_circle'], data['scale_circle'])
        self.contrastive_loss = ContrastiveLoss(data['margin_constrastive'], data['scale_constrast'])
        self.instance_loss = InstanceLoss(data['margin_instance'])

    def forward(self, embedding, targets):
        if self.norm_feat:
            dist_mat = euclidean_dist_fast_reid(F.normalize(embedding), F.normalize(embedding))
            # dist_mat = torch.matmul(F.normalize(embedding), F.normalize(embedding).T)
        else:
            dist_mat = euclidean_dist_fast_reid(embedding, embedding)

        N = dist_mat.size(0)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        if self.loss_function == "hard_mining":
            dist_ap, dist_an = hard_example_mining_fastreid(dist_mat, is_pos, is_neg)
        elif self.loss_function == "konwledge_mining":
            dist_ap, dist_an = acknowledge_example_mining(dist_mat, is_pos, is_neg)
        elif self.loss_function == "easy_positive":
            dist_ap, dist_an = easy_postive_mining(dist_mat, is_pos, is_neg)
        elif self.loss_function == "semi_hard_positive":
            dist_ap, dist_an = semi_hard_positive_mining(dist_mat, is_pos, is_neg, self.margin)
        elif self.loss_function == "easy_negative":
            dist_ap, dist_an = easy_negative_mining(dist_mat, is_pos, is_neg, self.margin)
        elif self.loss_function == "semi_hard_negative":
            dist_ap, dist_an = semi_hard_negtive_mining(dist_mat, is_pos, is_neg, self.margin)
        elif self.loss_function == "circle_loss":
            return self.circle_loss(embedding, targets)
        elif self.loss_function == "contrastive_loss":
            return self.contrastive_loss(embedding, targets)
        elif self.loss_function == 'instance_loss':
            return self.instance_loss(embedding, targets)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self.margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on

        return loss