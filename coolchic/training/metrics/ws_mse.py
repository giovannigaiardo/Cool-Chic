import torch
from torch import Tensor


def precompute_erp_weights(height, device="cpu", dtype=torch.float32):
    j = torch.arange(height, device=device, dtype=dtype)
    row_weights = torch.cos((j + 0.5 - height / 2.0) * torch.pi / height)
    return row_weights.view(1, 1, height, 1)


def ws_mse_fn(x: Tensor, y: Tensor, weight_map: Tensor) -> Tensor:
    weight_map = weight_map.to(device=x.device, dtype=x.dtype)
    square_error = (x - y).square()
    expanded_weights = weight_map.expand_as(square_error)

    return (square_error * expanded_weights).sum() / expanded_weights.sum()
