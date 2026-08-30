import torch
from torch import Tensor


def precompute_erp_weights(width, height, device="cpu", dtype=torch.float32):
    j = torch.arange(height, device=device, dtype=dtype)
    row_weights = torch.cos((j + 0.5 - height / 2.0) * torch.pi / height)
    return row_weights.view(1, 1, height, 1)


def ws_mse_fn(x: Tensor, y: Tensor, weight_map: Tensor) -> Tensor:
    square_error = (x - y).square()
    return (square_error * weight_map).sum() / weight_map.sum()
