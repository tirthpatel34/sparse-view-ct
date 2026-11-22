"""
A Radon transform (and its adjoint) that torch.func.jvp can differentiate.

deepinv's parallel-beam Radon transform is built on `grid_sample`, which has no
forward-mode autodiff rule in PyTorch, so

    torch.func.jvp(lambda x: op.A(x), (x,), (v,))

raises NotImplementedError on `grid_sampler_2d`. Any model that needs a
Jacobian-vector product through a CT operator hits this.

The Radon transform is LINEAR, so its derivative does not depend on where it is
evaluated. For a linear map L:

    forward :  L(x)
    backward:  L^T(grad)    (reverse mode, loss.backward())
    jvp     :  L(tangent)   (forward mode, torch.func.jvp)

Both derivative rules are therefore exact, not approximations. This file states
them directly with torch.autograd.Function.

    from deepinv.physics import Tomography
    from radon_jvp import radon, radon_T

    op = Tomography(angles=..., img_width=181, circle=False,
                    adjoint_via_backprop=True, ...)
    y  = radon(x, op)       # (B, 1, W, W)       -> (B, 1, n_det, n_angles)
    xb = radon_T(y, op)     # (B, 1, n_det, n_angles) -> (B, 1, W, W)

Use `adjoint_via_backprop=True`: deepinv's default inverse-Radon adjoint is not
the exact adjoint of A (dot-product test ~0.992), so backward() would not be the
true gradient.

Run `python radon_jvp.py` for a self-test (CPU, a few seconds).

Depends only on torch and deepinv.
"""

from __future__ import annotations

import torch
from torch import Tensor

# With adjoint_via_backprop=True, deepinv computes A^T by running autograd
# backwards through A, which costs ~30 MB per 256 x 256 sinogram. A acts on each
# image independently, so processing the batch in chunks gives the identical
# answer with a fraction of the peak memory.
CHUNK = 8


def _A(op, x: Tensor) -> Tensor:
    return torch.cat([op.A(c) for c in x.split(CHUNK)], dim=0)


def _AT(op, y: Tensor) -> Tensor:
    return torch.cat([op.A_adjoint(c) for c in y.split(CHUNK)], dim=0)


class _RadonA(torch.autograd.Function):
    """x -> A x, with backward A^T g and jvp A v."""

    @staticmethod
    def forward(x, op):
        with torch.no_grad():
            return _A(op, x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.op = inputs[1]

    @staticmethod
    def backward(ctx, g):
        with torch.no_grad():
            return _AT(ctx.op, g), None

    @staticmethod
    def jvp(ctx, x_t, _op_t):
        with torch.no_grad():
            return _A(ctx.op, x_t)


class _RadonAT(torch.autograd.Function):
    """y -> A^T y, with backward A g and jvp A^T v."""

    @staticmethod
    def forward(y, op):
        with torch.no_grad():
            return _AT(op, y)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.op = inputs[1]

    @staticmethod
    def backward(ctx, g):
        with torch.no_grad():
            return _A(ctx.op, g), None

    @staticmethod
    def jvp(ctx, y_t, _op_t):
        with torch.no_grad():
            return _AT(ctx.op, y_t)


def radon(x: Tensor, op) -> Tensor:
    """Forward projection A x, differentiable in both forward and reverse mode."""
    return _RadonA.apply(x, op)


def radon_T(y: Tensor, op) -> Tensor:
    """Back-projection A^T y, differentiable in both forward and reverse mode."""
    return _RadonAT.apply(y, op)


# ----------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------
def _selftest(width: int = 64, n_angles: int = 64, batch: int = 3) -> None:
    from deepinv.physics import Tomography

    torch.manual_seed(0)
    angles = 180.0 / n_angles * torch.arange(n_angles, dtype=torch.float32)
    op = Tomography(angles=angles, img_width=width, circle=False, normalize=False,
                    fan_beam=False, adjoint_via_backprop=True,
                    device="cpu", dtype=torch.float32)

    x = torch.rand(batch, 1, width, width)
    v = torch.randn_like(x)
    y = op.A(x)
    w = torch.randn_like(y)

    def rel(a, b):
        return float((a - b).norm() / b.norm())

    # 1. the plain operator really does fail under forward mode
    try:
        torch.func.jvp(lambda z: op.A(z), (x,), (v,))
        print("note: plain op.A survived torch.func.jvp in this PyTorch version")
    except (NotImplementedError, RuntimeError) as e:
        print(f"plain op.A under jvp fails as expected: {type(e).__name__}")

    # 2. jvp of radon vs central finite difference (exact for a linear map, up to fp32)
    _, jv = torch.func.jvp(lambda z: radon(z, op), (x,), (v,))
    eps = 1e-2
    fd = (op.A(x + eps * v) - op.A(x - eps * v)) / (2 * eps)
    e_jvp = rel(jv, fd)

    # 3. jvp of radon_T vs A^T applied to the tangent
    _, jvT = torch.func.jvp(lambda z: radon_T(z, op), (y,), (w,))
    e_jvpT = rel(jvT, op.A_adjoint(w))

    # 4. backward of radon: grad of <A x, w> w.r.t. x must equal A^T w
    xg = x.clone().requires_grad_(True)
    (radon(xg, op) * w).sum().backward()
    e_bwd = rel(xg.grad, op.A_adjoint(w))

    # 5. adjoint (dot-product) test: <A v, w> == <v, A^T w>
    lhs = float((radon(v, op) * w).sum())
    rhs = float((v * radon_T(w, op)).sum())
    e_adj = abs(lhs - rhs) / abs(rhs)

    print(f"jvp(radon)   vs finite difference : rel err {e_jvp:.2e}")
    print(f"jvp(radon_T) vs A^T v             : rel err {e_jvpT:.2e}")
    print(f"backward(radon) vs A^T g          : rel err {e_bwd:.2e}")
    print(f"adjoint test <Av,w> vs <v,A^T w>  : rel err {e_adj:.2e}")

    tol = 1e-4
    ok = max(e_jvp, e_jvpT, e_bwd, e_adj) < tol
    print("PASS" if ok else f"FAIL (tolerance {tol})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
