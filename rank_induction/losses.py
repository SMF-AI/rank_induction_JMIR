from typing import Union, List, Optional
from collections import Counter
from functools import partial

import torch


def get_pos_weight(
    train_labels: torch.Tensor,
) -> torch.Tensor:
    """
    학습 데이터의 클래스 레이블 분포를 기반으로 벨런스 가중치를 반환

    Args:
        train_labels (torch.Tensor]): 학습 데이터에 포함된 레이블

    Returns:
        torch.Tensor: 각 클래스에 대한 가중치를 담은 PyTorch 텐서.

    Raises:
        ValueError: train_labels에 포함된 클래스 수가 label_class의 클레스 수와 일치하지않는 경우
    """
    if train_labels.numel() == 0:
        raise ValueError("train_labels are empty.")

    num_positive = train_labels.sum().item()
    num_total = len(train_labels)
    if num_positive == 0:
        raise ValueError("Positive cases are empty.")

    pos_weight = num_total / num_positive
    return torch.tensor(pos_weight, dtype=torch.float)

def ranknet_loss(
    y_prob: torch.Tensor,
    y_true: torch.Tensor,
    sigma=4.0,
    ignore_equal=True,
    margin: float = 0.0,
    device: str = "cuda",
    num_pos: int = None,
    num_neg: int = None,
) -> torch.Tensor:
    """RankNet의 손실 함수 값을 계산

    See Also:
        https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf

    Args:
        y_prob (torch.Tensor): instance probablity
        y_true (torch.Tensor): instance label
        sigma (float, optional): scale parameter of sigmoid func. Defaults to 1.0.
        ignore_equal (bool)
        margin (float, optional): Defaults to 0.0.
        device (str, optional): Defaults to "cuda".
        num_pos (int, optional): number of pos sample
        num_neg (int, optional): number of neg sample

    Returns:
        loss: rank loss in [0, 1]

    Note:
        1. ignore_eqaul=True 권장

    Example:
        >>> # 기준
        >>> y_prob = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> ranknet_loss(y_prob, y_true)
        tensor(0.6372)

        >>> # y_true 2개인 경우
        >>> y_prob = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        >>> ranknet_loss(y_prob, y_true)
        tensor(0.6011)

        >>> # y_true 2개인 경우 + 하나 밀린 경우
        >>> y_prob = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        >>> ranknet_loss(y_prob, y_true)
        tensor(0.6511)

        >>> # y_true 2개인 경우 + 랭크가 2개 밀린 경우
        >>> y_prob = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        >>> ranknet_loss(y_prob, y_true)
        tensor(0.6678)

        >>> # y_true 2개인 경우 + 랭크가 많이 밀린 경우
        >>> y_prob = torch.tensor([0.7, 0.5, 0.2, 0.1, 0.1, 0.1])
        >>> y_true = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        >>> ranknet_loss(y_prob, y_true)
        tensor(0.7678)
    """
    if num_pos is not None or num_neg is not None:
        pos_indices = torch.where(y_true == 1)[1]
        neg_indices = torch.where(y_true == 0)[1]
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_count = min(len(pos_indices), num_pos)
        pos_perm = torch.randperm(len(pos_indices), device=device)[:pos_count]
        pos_indices = pos_indices[pos_perm]
        
        neg_count = min(len(neg_indices), num_neg)
        neg_perm = torch.randperm(len(neg_indices), device=device)[:neg_count]
        neg_indices = neg_indices[neg_perm]

        chosen_indices = torch.cat([pos_indices, neg_indices], dim=0)
        y_prob = y_prob[:, chosen_indices] # (1, N)
        y_true = y_true[:, chosen_indices] # (1, N)
        
    diff_matrix = y_prob.view(-1, 1) - y_prob.view(1, -1) - margin 
    pij = torch.sigmoid(sigma * diff_matrix)

    # Y True와 차이를 계산하여 Pbar을 구함
    label_diff = y_true.view(-1, 1) - y_true.view(1, -1)
    pbar = torch.where(
        label_diff > 0,
        torch.tensor(1.0, device=device),
        torch.where(
            label_diff == 0,
            torch.tensor(0.5, device=device),
            torch.tensor(0.0, device=device),
        ),
    )

    if ignore_equal:
        mask = torch.where(pbar != 0.5)
        pbar = pbar[mask]
        pij = pij[mask]

    pij = pij.ravel()
    pbar = pbar.ravel()
    return torch.nn.functional.binary_cross_entropy(pij, pbar)


class AttentionInductionLoss(torch.nn.Module):
    def __init__(self, _lambda: float, pos_weight=Optional[torch.Tensor]):
        super().__init__()
        self._lambda = _lambda
        self.pos_weight = pos_weight
        self.bag_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.attention_loss = torch.nn.MSELoss(reduction='sum')

    def forward(
        self,
        bag_true: torch.Tensor,
        bag_pred: torch.Tensor,
        instance_true: torch.Tensor,
        instance_pred: torch.Tensor,
    ) -> torch.Tensor:

        bag_loss = self.bag_loss(bag_pred, bag_true)

        if bag_true.item() == 1:
            attention_loss = self.attention_loss(instance_pred, instance_true)
            return bag_loss + self._lambda * attention_loss

        return bag_loss


class RankNetLoss(torch.nn.Module):
    def __init__(
        self,
        _lambda: float,
        sigma: float = 1.0,
        ignore_equal: bool = False,
        margin: float = 0.0,
        num_pos: int = None,
        num_neg: int = None,
        device: str = "cuda",
    ):
        super().__init__()
        self._lambda = _lambda
        self.sigma = sigma
        self.ignore_equal = ignore_equal
        self.margin = margin
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.device = device
        self.bag_loss = torch.nn.BCEWithLogitsLoss()
        self.attention_loss = partial(
            ranknet_loss,
            sigma=self.sigma,
            ignore_equal=self.ignore_equal,
            margin=self.margin,
            num_pos=self.num_pos,
            num_neg=self.num_neg,
            device=self.device,
        )

    def forward(
        self,
        bag_true: torch.Tensor,
        bag_pred: torch.Tensor,
        instance_true: torch.Tensor,
        instance_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
            RankNet loss의 경우, bag_true가 1이고 instance_true가 0이 아닌 경우에만 
            attention loss를 추가로 계산해줌.
            
            Note:
            1) bag_ture가 1인데, instance_true가 0인경우: Fraction of expert-annotation
        """
        bag_loss = self.bag_loss(bag_pred, bag_true)
        if bag_true.item() == 1 and instance_true.sum().item() > 0:
            attention_loss = self.attention_loss(instance_pred, instance_true)
            return bag_loss + self._lambda * attention_loss
        else:
            return bag_loss
