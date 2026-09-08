"""Ridge/CCA-normalised HSIC (nHSIC) used by the paper Predictor.

Adapted from the official HSIC-bottleneck implementation
https://github.com/choasma/HSIC-bottleneck (MIT).
The estimator is Ma, Lewis, and Kleijn, "The HSIC Bottleneck:
Deep Learning without Back-Propagation", AAAI 2020. It is not a new
dependence measure.

For features X, Y in R^{B x d}, with centered Gaussian Gram matrices K, L:

    nHSIC(X, Y) = < (K + eps B I)^{-1} K ,  ((L + eps B I)^{-1} L)^T >

When ``sigma`` is None (paper training), the RBF bandwidth is the median
pairwise distance, clamped at 1e-2, then scaled by feature dimension.
"""

import torch


def kernelmat_fast(x, sigma=None):
    """Centered RBF Gram matrix of a minibatch ``x`` (B x d)."""
    batch_size = x.size(0)
    squared_distance = torch.cdist(x, x, p=2) ** 2

    if sigma is None:
        lower_triangle = torch.tril_indices(batch_size, batch_size, offset=-1)
        distances = squared_distance.flatten()[lower_triangle]
        sigma = torch.clamp(torch.median(distances), min=1e-2).item()

    variance = 2.0 * sigma * sigma * x.size(1)
    kernel = torch.exp(-squared_distance / variance)
    return (
        kernel
        - kernel.mean(0, keepdim=True)
        - kernel.mean(1, keepdim=True)
        + kernel.mean()
    )


def hsic_normalized_cca_fast(x, y, sigma=None, epsilon=1e-5):
    """Scalar nHSIC dependence between two minibatches of equal length."""
    if x.dim() == 1:
        x = x.view(-1, 1)
    if y.dim() == 1:
        y = y.view(-1, 1)
    if x.size(0) != y.size(0):
        raise ValueError("x and y must contain the same number of samples")

    batch_size = x.size(0)
    kernel_x = kernelmat_fast(x, sigma)
    kernel_y = kernelmat_fast(y, sigma)
    ridge = epsilon * batch_size * torch.eye(
        batch_size, device=x.device, dtype=x.dtype
    )

    projection_x = torch.linalg.solve(kernel_x + ridge, kernel_x)
    projection_y = torch.linalg.solve(kernel_y + ridge, kernel_y)
    return torch.sum(projection_x * projection_y.T)
