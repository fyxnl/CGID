r"""Functional interface"""
from typing import Callable, List, Optional, Tuple
import math
import warnings

import torch
from torch import _VF
from torch._C import _infer_size, _add_docstr
# from torch._torch_docs import reproducibility_notes, tf32_notes

from torch._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
# from torch.overrides import (
    # has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    # handle_torch_function)
from torch.overrides import (
    has_torch_function,
    handle_torch_function)
from torch.nn import _reduction as _Reduction
from torch.nn import grad  # noqa: F401
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default


Tensor = torch.Tensor

def gaussian_loss(
    input: Tensor,
    target: Tensor,
    # var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    r"""Gaussian negative log likelihood loss.

    See :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            gaussian_loss,
            (input, target),
            input,
            target,

            full=full,
            eps=eps,
            reduction=reduction,
        )

    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    # if var.size() != input.size():
    #
    #     # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
    #     # e.g. input.size = (10, 2, 3), var.size = (10, 2)
    #     # -> unsqueeze var so that var.shape = (10, 2, 1)
    #     # this is done so that broadcasting can happen in the loss calculation
    #     if input.size()[:-1] == var.size():
    #         var = torch.unsqueeze(var, -1)
    #
    #     # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
    #     # This is also a homoscedastic case.
    #     # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
    #     elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
    #         pass
    #
    #     # If none of the above pass, then the size of var is incorrect.
    #     else:
    #         raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    # if torch.any(var < 0):
    #     raise ValueError("var has negative entry/entries")
    #
    # # Clamp for stability
    # var = var.clone()
    # with torch.no_grad():
    #     var.clamp_(min=eps)

    # Calculate the loss
    # loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
    loss = 0.5 * math.pi * torch.exp(-torch.abs(input - target) ** 2 / 2*0.05)
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :class:`~torch.nn.MSELoss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            mse_loss, (input, target), input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))

def  meeloss(input, target, size_average=None, reduce=None, reduction='mean'):
    if not torch.jit.is_scripting():
        tens_ops = (input, target)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                mse_loss, tens_ops, input, target, size_average=size_average, reduce=reduce,
                reduction=reduction)
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = torch.exp(-torch.abs(input - target) ** 2)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret