"""
UNINA-DLA Custom Optimizers.

Provides Muon and MuSGD (Muon + SGD hybrid) optimizer implementations
with matrix orthogonalization via Newton-Schulz iterations.
"""

import torch
from torch import optim


def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute the zeroth power / orthogonalization of matrix G using Newton-Schulz iteration."""
    assert len(G.shape) == 2
    X = G.to(torch.bfloat16)
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for a, b, c in [  # num_steps fixed at 5
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
    ]:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


def muon_update(
    grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, nesterov: bool = True
) -> torch.Tensor:
    """Compute Muon optimizer update with momentum and orthogonalization."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class MuSGD(optim.Optimizer):
    """
    Hybrid optimizer combining Muon and SGD (MuSGD) for YOLO26 training.

    Implements the official YOLO26 optimizer approach:
    - Muon: Matrix orthogonalization via Newton-Schulz iterations for 2D weights
    - SGD: Nesterov momentum for auxiliary parameters (biases, norms)

    Based on "Muon is Scalable for LLM Training" (arXiv:2502.16982) and adapted
    for computer vision by Ultralytics YOLO26. Provides ~43% faster convergence.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon_ratio: float = 0.5,
        sgd_ratio: float = 0.5,
        muon: float | None = None,
        sgd: float | None = None,
        **kwargs,
    ):
        if muon is not None:
            muon_ratio = muon
        if sgd is not None:
            sgd_ratio = sgd
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
            **kwargs,
        )
        super().__init__(params, defaults)
        self.muon_ratio = muon_ratio
        self.sgd_ratio = sgd_ratio

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon
            if group["use_muon"]:
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_SGD"] = torch.zeros_like(p)

                    # Compute weight-decayed gradient on current parameter p_t before in-place Muon update
                    grad_sgd = (
                        grad.add(p, alpha=group["weight_decay"])
                        if group["weight_decay"] != 0
                        else grad
                    )

                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=group["nesterov"],
                    )
                    p.add_(update.reshape(p.shape), alpha=-(lr * self.muon_ratio))

                    # SGD update
                    state["momentum_buffer_SGD"].mul_(group["momentum"]).add_(grad_sgd)
                    sgd_update = (
                        grad_sgd.add(state["momentum_buffer_SGD"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_SGD"]
                    )
                    p.add_(sgd_update, alpha=-(lr * self.sgd_ratio))
            else:  # SGD
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    p.add_(update, alpha=-lr)
        return loss
