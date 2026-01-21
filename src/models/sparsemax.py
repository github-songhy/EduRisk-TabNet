"""稀疏化激活函数。"""

from __future__ import annotations

import torch


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """稀疏化softmax实现。"""

    if logits.numel() == 0:
        return logits

    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)

    dim_size = z.size(dim)
    range_tensor = torch.arange(1, dim_size + 1, device=logits.device, dtype=logits.dtype)
    view_shape = [1] * z.dim()
    view_shape[dim] = dim_size
    range_tensor = range_tensor.view(view_shape)

    support = 1 + range_tensor * z_sorted > z_cumsum
    support_sum = support.sum(dim=dim, keepdim=True)
    support_sum = support_sum.clamp(min=1)

    z_support = z_cumsum.gather(dim, support_sum - 1)
    tau = (z_support - 1) / support_sum.to(logits.dtype)

    output = torch.clamp(z - tau, min=0)
    return output
