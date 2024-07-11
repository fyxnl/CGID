import torch.nn as nn
import math
import torch
import warnings
import torch.nn.functional as F
from torch import Tensor
from torch import Tensor
from typing import Callable, Optional
# import gaussian_loss
from gaussian_loss import gaussian_loss,meeloss
import torch.nn._reduction as _Reduction
class _Loss(nn.Module):
    reduction: str
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
def legacy_get_string(size_average, reduce, emit_warning=True):
    # type: (Optional[bool], Optional[bool], bool) -> str
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret

# class _WeightedLoss(_Loss):
#     def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
#         super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
#         self.register_buffer('weight', weight)
#         self.weight: Optional[Tensor]

class MEELoss(_Loss):
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float
    def __init__(self, *,  full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(MEELoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return gaussian_loss(input, target,  full=self.full, eps=self.eps, reduction=self.reduction)
class MEELoss1(_Loss):
    __constants__ = ['reduction']
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MEELoss1, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return meeloss(input, target, reduction=self.reduction)





