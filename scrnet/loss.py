import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class SelectiveLoss(nn.Module):
    def __init__(self, start: int, length: int,weight, alpha: float = 0.1, use_mse=True):
        """
        start: 合法 label 的起始 index
        length: 合法 label 的数量
        alpha: 非法 logits 的惩罚项权重
        use_mse: 如果为 True 使用 MSE loss，否则使用 L2 norm loss
        """
        super().__init__()
        self.weight=weight
        self.start = start
        self.end = start + length
        self.alpha = alpha
        self.use_mse = use_mse

        # self.low_acc_idx = torch.from_numpy(np.load("low_acc_idx.npy"))
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.low_acc_idx = self.low_acc_idx.to(device)


    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [batch_size, num_classes]
        labels: [batch_size]
        """
        valid_mask = (labels >= self.start) & (labels < self.end)
        # valid_mask = torch.isin(labels, self.low_acc_idx)
        invalid_mask = ~valid_mask
        # print (valid_mask)
        loss_valid = torch.tensor(0.0, device=logits.device)
        loss_invalid = torch.tensor(0.0, device=logits.device)

        # 1. 合法标签的 CrossEntropyLoss
        if valid_mask.any():
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask] # 偏移标签
            loss_valid = F.cross_entropy(valid_logits, valid_labels)

        # # 2. 非法 logits 的惩罚项：趋近于 0 向量
        if invalid_mask.any():
            invalid_logits = logits[invalid_mask]
            target = torch.full_like(invalid_logits, fill_value=0.5 , device=logits.device)
            if self.use_mse:
                loss_invalid = F.mse_loss(invalid_logits, target)
            else:
                loss_invalid = torch.mean(torch.norm(invalid_logits, dim=1))  # L2 norm

        return loss_valid + loss_invalid
if __name__=='__main__':
    a = torch.tensor([1,2,3])
    valid_mask = (a >= 1) & (a < 3)
    print (valid_mask)