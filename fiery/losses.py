import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255, future_discount=1.0):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index
        self.future_discount = future_discount

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        seq_len = loss.shape[1]
        future_discounts = self.future_discount ** torch.arange(seq_len, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, seq_len, 1, 1, 1)
        loss = loss * future_discounts

        return loss[mask].mean()


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False, top_k_ratio=1.0, future_discount=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

    def forward(self, prediction, target):
        if target.shape[-3] != 1:
            raise ValueError('segmentation label must be an index-label with channel dimension = 1.')
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )

        loss = loss.view(b, s, h, w)

        future_discounts = self.future_discount ** torch.arange(s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)


class ProbabilisticLoss(nn.Module):
    def forward(self, output):
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
            present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return kl_loss


# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    ignore_index: int = -100,
    reduction: str = 'mean',
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C-1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        ignore_index: If ignore_index is specified, this loss also accepts this class index (this index may not necessarily be in the class range).
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) < 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if input.device != target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    target = target.unsqueeze(1)
    src = torch.ones_like(target, dtype=input.dtype)
    src[target == ignore_index] = 0
    target_one_hot = torch.zeros_like(input)
    target_one_hot.scatter_(dim=1, index=target, src=src)

    # compute softmax over the classes axis
    input_soft = F.softmax(input, dim=1)
    log_input_soft = F.log_softmax(input, dim=1)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.sum(loss_tmp) / (target != ignore_index).sum()
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        ignore_index: If ignore_index is specified, this loss also accepts this class index (this index may not necessarily be in the class range).
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C-1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.ignore_index, self.reduction)


if __name__ == '__main__':
    N = 5  # num_classes
    input = torch.randn(3, N, requires_grad=True)
    # input = torch.randn(1, N, 3, 5, requires_grad=True)
    # target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
    target = torch.empty(3, dtype=torch.long).random_(N)
    output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='none', ignore_index=1)
    # print(output.shape)
    output = output.mean()
    output.backward()
