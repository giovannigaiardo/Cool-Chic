import torch
import torch.nn.functional as F
from torch import Tensor


def ssim_fn(
    x: Tensor,
    y: Tensor,
    weight_map: Tensor = None,
    k: int = 11,
    gaussian: bool = True,
    sigma: float = 1.5,
    max_val: float = 1.0,
) -> Tensor:
    device, dtype = x.device, x.dtype
    _, C, H, W = x.shape

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    if gaussian:
        coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        g = torch.exp(-coords.square() / (2 * sigma**2))
        g /= g.sum()
        kernel_2d = g.view(-1, 1) @ g.view(1, -1)
    else:
        kernel_2d = torch.full((k, k), 1.0 / (k**2), device=device, dtype=dtype)

    kernel = kernel_2d.expand(C, 1, -1, -1).contiguous()

    padding = k // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=C)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=C)

    mu_x_square = mu_x.square()
    mu_y_square = mu_y.square()
    mu_x_mu_y = mu_x * mu_y

    sigma_x_square = (
        F.conv2d(x.square(), kernel, padding=padding, groups=C) - mu_x_square
    )
    sigma_y_square = (
        F.conv2d(y.square(), kernel, padding=padding, groups=C) - mu_y_square
    )
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=C) - mu_x_mu_y

    numerator = (2 * mu_x_mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_square + mu_y_square + c1) * (
        sigma_x_square + sigma_y_square + c2
    )
    ssim_map = numerator / denominator

    center_ssim = ssim_map[:, :, padding : H - padding, padding : W - padding]
    if weight_map is None:
        return center_ssim.mean()

    weight_map = weight_map.to(device)

    center_weights = weight_map[:, :, padding : H - padding, :]

    broadcasted_weight = center_weights.expand_as(center_ssim)
    weighted_ssim_sum = (center_ssim * broadcasted_weight).sum(dim=(2, 3))
    total_weights_sum = broadcasted_weight.sum(dim=(2, 3))

    return (weighted_ssim_sum / total_weights_sum).mean()
