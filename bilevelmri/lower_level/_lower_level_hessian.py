import torch


def regulariser_hessian(x, w, A, reg_func):
    Ax, Aw = A(x), A(w)
    return A.T(reg_func.hess(Ax, Aw))


def lower_level_hessian(x, w, S, alpha, eps, A, reg_func):
    # Compute the action of the lower level Hessian
    hess_data = torch.ifft(
        S.view(1, *S.shape, 1)**2 * torch.fft(
            w, signal_ndim=2, normalized=True),
        signal_ndim=2,
        normalized=True)
    hess_reg = regulariser_hessian(x, w, A, reg_func)
    return hess_data + alpha * hess_reg + eps * w


def lower_level_mixed_derivs(x, w, y, S, alpha, eps, A, reg_func):
    # Compute mixed second derivatives of the lower level objective function for use in the adjoint method
    Fw = torch.fft(w, 2, normalized=True)
    DwDy = -S.view(1, *S.shape, 1)**2 * Fw
    DwDalpha = torch.sum(w * A.T(reg_func.grad(A(x))))
    Fx = torch.fft(x, 2, normalized=True)
    DwDS = torch.sum(Fw * 2 * S.view(1, *S.shape, 1) * (Fx - y), dim=(0, 3))
    DwDeps = torch.sum(w * x)
    return DwDy, DwDS, DwDalpha, DwDeps
