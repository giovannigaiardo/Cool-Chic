# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


from enum import Flag, auto

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch import Tensor, nn
from torchvision.models.vgg import VGG16_Weights

"""
Code inspired by the Codex implementation of the Wasserstein Distance, available
at: https://github.com/google/codex/blob/main/codex/loss/wasserstein.py
"""


class SphericalMode(Flag):
    """Which spherical (ERP) mechanisms are active in the Wasserstein distortion.

    The two mechanisms are independent, so this is a ``Flag`` and they can be
    combined. This enables a clean 2x2 ablation:

        NONE           no spherical mechanism (constant log2_sigma + plain mean)
        COSWEIGHT      only the cos(phi) spatial pooling weight
        SIGMA          only the latitude-driven (equatorial-bias) log2_sigma map
        BOTH           both mechanisms (SIGMA | COSWEIGHT)
    """

    NONE = 0
    SIGMA = auto()
    COSWEIGHT = auto()
    BOTH = SIGMA | COSWEIGHT


def safe_clamp_min(x: Tensor, min: float = 0.0) -> Tensor:
    """Clamp a tensor so that its minimum value correspond to <min>. But
    let an identity gradient flow i.e.

        y = safe_clamp_min(x, 0.) --> dy / dx = 1

    Args:
        x (Tensor): Tensor to be clamped
        min (float, optional): Minimal value for the clamp. Defaults to 0..

    Returns:
        Tensor: Clamped value
    """
    # Forward point-of-view: y = x - x + clamp(x, min) = clamp(x, min)
    # Backward point-of-view: y = x --> dy/dx = 1
    y = x
    with torch.no_grad():
        y = y - x + torch.clamp_min(x, min=min)
    return y


class Vgg16(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Select features right before MaxPool2d. I could add 29 to the list
        self.desired_ft = [3, 8, 15, 22]

        # I don't need to go through the last layer after the desired features
        features = list(
            torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        )[: max(self.desired_ft) + 1]
        self.device = "cpu"
        self.features = torch.nn.ModuleList(features).eval()

        self.to_device(self.device)

    def forward(self, x):
        results = []
        for idx_ft, lay in enumerate(self.features):
            x = lay(x)
            if idx_ft in self.desired_ft:
                results.append(rearrange(x, "b c h w -> (b c) 1 h w", b=1))

        return results

    def to_device(self, device) -> None:
        self.features = self.features.to(device)
        self.device = device


class WassersteinLoss(nn.Module):
    def __init__(
        self,
        mode: SphericalMode = SphericalMode.NONE,
        sigma_min: float = 3.0,
        sigma_max: float = 24.0,
        sigma_gaze_deg: float = 20.0,
    ):
        super().__init__()
        self.wd_net = Vgg16()
        self.device = "cpu"
        self.lowpass_2d_kernel = torch.outer(
            torch.tensor([0.25, 0.5, 0.25]), torch.tensor([0.25, 0.5, 0.25])
        ).view(1, 1, 3, 3)
        self.target_features = None

        self.mode = mode
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_gaze_deg = sigma_gaze_deg

    def to_device(self, device) -> None:
        self.wd_net = self.wd_net.to(device)
        self.lowpass_2d_kernel = self.lowpass_2d_kernel.to(device)

        if self.target_features is not None:
            self.target_features = [tmp.to(device) for tmp in self.target_features]

        self.device = device

    def forward(self, decoded_img: Tensor, target_img: Tensor) -> Tensor:
        decoded_ft = self.wd_net(decoded_img)
        if self.target_features is None:
            target_ft = self.wd_net(target_img)
            self.target_features = target_ft
        else:
            target_ft = self.target_features
        wasserstein_distortion = self.multi_wasserstein_distortion(
            decoded_ft, target_ft, target_img.size()[-2:]
        )
        return wasserstein_distortion

    def multi_wasserstein_distortion(
        self,
        decoded_features: list[Tensor],
        target_features: list[Tensor],
        img_size: tuple[int, int],
        log2_sigma: int = 3,
        num_levels: int = 5,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Wasserstein Distortion between multiple feature arrays of two images.

        This function accepts more than one feature array per image. The arrays don't
        need to be all the same shape, but the nth array in `decoded_features` must have the
        same shape as the nth array in `target_features`. The aspect ratio of all the
        arrays should be approximately the same.

        Args:
        decoded_features: Multiple feature arrays of format `(channels, height, width)`,
            corresponding to the first image to be compared.
        target_features: Multiple feature arrays of format `(channels, height, width)`,
            corresponding to the second image to be compared.
        img_size: (Height, Width) of the image.
        log2_sigma: Array, shape `(height, width)`. The base two logarithm of the
            sigma map, which indicates the amount of summarization in each location.
            Doesn't have to have the same shape as the feature arrays.
        num_levels: Integer. The number of multi-scale levels of the feature
            statistics to compute. Must be greater or equal to the maximum of
            `log2_sigma`.

        Returns:
        Distortion value
        """
        if len(decoded_features) != len(target_features):
            raise ValueError(
                f"`decoded_features` and `target_features` must have same length, but received "
                f"{len(decoded_features)} and {len(target_features)}, respectively."
            )

        device = decoded_features[0].device

        dist = torch.zeros((1), device=device)
        for fa, fb in zip(decoded_features, target_features):
            if fa.size() != fb.size():
                raise ValueError(
                    f"Found feature arrays with incompatible sizes. A: {fa.size()}, B: {fb.size()}."
                )

            if SphericalMode.SIGMA in self.mode:
                ls = self.latitude_log2_sigma(fa)
            else:
                # For now: log2_sigma is constant
                ls = torch.ones((1, 1, *fa.size()[-2:]), device=device) * log2_sigma

            # This is aligned with the original codex implementation. It's been found
            # to be slightly worse than just letting the ls to log2_sigma for all
            # features as above.

            # # Rescale sigma to match the feature arrays. For example, if a feature array
            # # has a very low spatial resolution, we make sigma correspondingly smaller,
            # # because each element in the feature array covers a larger portion of the
            # # image. Since we are in log space, we subtract the log of the size ratio and
            # # then cap at zero.
            # # The initial sigma resolution is aligned with the image size.
            # log_ratio_h = np.log2(img_size[-2] / fa.shape[-2])
            # log_ratio_w = np.log2(img_size[-1] / fa.shape[-1])
            # mean_log_ratio = (log_ratio_h + log_ratio_w) / 2
            # ls = F.relu(ls - mean_log_ratio)

            d = self.wasserstein_distortion(
                fa,
                fb,
                ls,
                num_levels=num_levels,
            )
            dist = dist + d

        return dist

    def latitude_log2_sigma(self, feature_array: Tensor) -> Tensor:
        fh, fw = feature_array.size()[-2:]
        device, dtype = feature_array.device, feature_array.dtype
        # cos(phi) per row, shape (1, 1, fh, 1)
        j = torch.arange(fh, device=device, dtype=dtype)
        phi = (j + 0.5 - fh / 2.0) * (torch.pi / fh)
        cos_phi = torch.cos(phi)

        sigma_gaze_rad = torch.tensor(
            self.sigma_gaze_deg * torch.pi / 180.0, device=device, dtype=dtype
        )
        gaze_gaussian = torch.exp(-(phi**2) / (2.0 * sigma_gaze_rad**2))
        s = cos_phi * gaze_gaussian

        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-7)

        sigma = self.sigma_max - s_norm * (self.sigma_max - self.sigma_min)
        log2_sigma_rows = torch.log2(sigma)

        return log2_sigma_rows.view(1, 1, fh, 1).expand(1, 1, fh, fw).contiguous()

    def wasserstein_distortion(
        self,
        decoded_features: Tensor,
        target_features: Tensor,
        log2_sigma: Tensor,
        num_levels: int = 5,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Evaluates Wasserstein Distortion between two feature arrays.

        Args:
        decoded_features: Array, shape `(channels, height, width)`. The first feature
            array to be compared.
        target_features: Array, shape `(channels, height, width)`. The second feature
            array to be compared.
        log2_sigma: Array, shape `(height, width)`. The base two logarithm of the
            sigma map, which indicates the amount of summarization in each location.
            Must have the same height and width as the feature arrays.
        num_levels: Integer. The number of multi-scale levels of the feature
            statistics to compute. Must be greater or equal to the maximum of
            `log2_sigma`.
        sqrt_grad_limit: Float. Upper limit for the gradient of the square root
            applied to the empirical feature variance estimates, for numerical
            stability.
        return_intermediates: Boolean. If `True`, returns intermediate computations
            in a dictionary, besides the distortion value.

        Returns:
        Distortion value
        """
        if decoded_features.shape != target_features.shape:
            raise ValueError(
                f"`decoded_features` and `target_features` must have same shape, but received "
                f"{decoded_features.shape} and {target_features.shape}, respectively."
            )

        means_a, variances_a = self.compute_multiscale_stats(
            decoded_features, num_levels
        )
        means_b, variances_b = self.compute_multiscale_stats(
            target_features, num_levels
        )

        assert len(means_a) == len(means_b) == len(variances_a) == len(variances_b)

        wd_maps = [torch.square(decoded_features - target_features)]
        for ma, mb, va, vb in zip(means_a, means_b, variances_a, variances_b):
            assert ma.shape == mb.shape == va.shape == vb.shape
            # Variance estimates can turn out slightly negative due to numerics. This
            # brings such estimates up to zero, but passes through a useful gradient.

            # In codex implementation, the sqrt derivative is clamped to
            # sqrt_grad_limit=1e6 by default. Since f'(x) = d sqrt(x) / dx = 1/2 * 1 / sqrt(x),
            # Having f'(x) >= 1e6 --> x <= 5e-7. A simpler way to clamp the gradient of
            # sqrt(x) is simply to clamp the variance so that it is **always** bigger than
            # 5e-7
            va = safe_clamp_min(va, min=5e-7)
            vb = safe_clamp_min(vb, min=5e-7)
            # The square root has unbounded gradients near zero. This limits the
            # gradient to a finite value.
            sa = torch.sqrt(va)
            sb = torch.sqrt(vb)
            wd_maps.append(torch.square(ma - mb) + torch.square(sa - sb))
        assert len(wd_maps) == num_levels + 1

        dist = torch.zeros((1), device=decoded_features.device)
        # intermediates = collections.defaultdict(list)
        # intermediates.update(wd_maps=wd_maps)
        for i, wd_map in enumerate(wd_maps):
            assert wd_map.size()[-2:] == log2_sigma.size()[-2:]
            weight = F.relu(1 - abs(log2_sigma - i))
            # intermediates["weights"].append(weight)
            if SphericalMode.COSWEIGHT in self.mode:
                h_curr = wd_map.size(-2)
                j = torch.arange(h_curr, device=wd_map.device, dtype=wd_map.dtype)
                phi = (j + 0.5 - h_curr / 2.0) * (torch.pi / h_curr)
                cos_phi = torch.cos(phi).view(1, 1, h_curr, 1)

                cos_weight = cos_phi / torch.mean(cos_phi)
                dist = dist + torch.mean(weight * wd_map * cos_weight)
            else:
                dist = dist + torch.mean(weight * wd_map)

            if i > 0:
                log2_sigma = self.lowpass(log2_sigma, stride=2)

        # if return_intermediates:
        #     return dist, intermediates
        return dist

    def lowpass(self, x: Tensor, stride: int) -> Tensor:
        """Lowpass filters an array of shape (batch, 1, height, width).

        Args:
        x: The input array of shape (batch, 1, height, width).
        stride: The stride length of the convolution. Typically either 1 or 2.

        Returns:
        The lowpass filtered array of shape (batch, 1, height, width). Height and width
        are the same as the input array if stride is 1.
        """

        # 3x3 kernel so padding of size 1
        return F.conv2d(x, self.lowpass_2d_kernel, stride=stride, padding=1)

    def compute_multiscale_stats(
        self,
        features: Tensor,
        num_levels: int,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Computes local mean and variance of a feature array."""
        squared = torch.square(features)
        means = []
        variances = []
        for _ in range(num_levels):
            m = self.lowpass(features, stride=1)
            p = self.lowpass(squared, stride=1)
            means.append(m)
            variances.append(p - torch.square(m))
            features = m[..., ::2, ::2]
            squared = p[..., ::2, ::2]
        return means, variances


# One WassersteinLoss module is built (and cached) per SphericalMode so that
# the VGG network is instantiated only once per mode.
_wasserstein_modules: dict["SphericalMode", "WassersteinLoss"] = {}


def wasserstein_fn(
    decoded_img: Tensor,
    target_img: Tensor,
    spherical: bool = False,
    mode: "SphericalMode | None" = None,
) -> Tensor:
    """Compute the wasserstein distance between two [1, 3, H, W] tensors.
    Dynamic range of the tensors is assumed to be [0., 1.]

    Args:
        decoded_img: Compressed image
        target_img: Reference image
        spherical: Backward-compatible shortcut. When ``mode`` is not given,
            ``spherical=True`` maps to ``SphericalMode.BOTH`` and
            ``spherical=False`` maps to ``SphericalMode.NONE``.
        mode: Which spherical (ERP) mechanisms to enable. Takes precedence over
            ``spherical`` when provided. See :class:`SphericalMode`.

    Returns:
        Tensor: Wasserstein distance (scalar value).
    """

    if mode is None:
        mode = SphericalMode.BOTH if spherical else SphericalMode.NONE

    module = _wasserstein_modules.get(mode)
    if module is None:
        module = torch.compile(WassersteinLoss(mode=mode), disable=True)
        _wasserstein_modules[mode] = module

    if decoded_img.device != module.device:
        module.to_device(decoded_img.device)

    return module(decoded_img, target_img)
