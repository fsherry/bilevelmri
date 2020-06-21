from .functionals import Functional
from ..linear_ops.wavelet_coefficients import WaveletCoefficients
import torch


def rho(x, gamma):
    out = torch.zeros_like(x)
    ind = x <= gamma
    out[ind] = -x[ind]**3 / (3 * gamma**2) + x[ind]**2 / gamma
    ind = x > gamma
    out[ind] = x[ind] - gamma / 3
    return out


def phi(x, gamma):
    out = torch.zeros_like(x)
    ind = x <= gamma
    out[ind] = -x[ind] / gamma**2 + 2 / gamma
    ind = x > gamma
    out[ind] = 1. / x[ind]
    return out


def prox_rho(norm_x, gamma, tau):
    if tau == 0.:
        return norm_x
    else:
        C = torch.zeros_like(norm_x)
        ind = norm_x <= gamma + tau
        C[ind] = (gamma / tau) * ((tau + 0.5 * gamma) - torch.sqrt(
            (tau + 0.5 * gamma)**2 - tau * norm_x[ind]))
        ind = norm_x > gamma + tau
        C[ind] = norm_x[ind] - tau
        return C


def psi(x, gamma):
    out = torch.zeros_like(x)
    ind = (x > 0) * (x <= gamma)
    out[ind] = -1. / (gamma**2 * x[ind])
    ind = x > gamma
    out[ind] = -1. / x[ind]**3
    return out


class Smoothed1Norm(Functional):
    def __init__(self, gamma=1e-2):
        super().__init__()
        self.gamma = gamma
        self.smoothness_bound = 4. / gamma

    def __call__(self, z):
        # wavelet coefficients are given in the form of a tuple of a torch tensor and a list of torch tensors
        if isinstance(z, WaveletCoefficients):
            z = z.coefs
            norm = torch.sum(
                rho(torch.sqrt(torch.sum(z[0]**2, dim=1)), self.gamma))
            for coefs in z[1]:
                norm += torch.sum(
                    rho(torch.sqrt(torch.sum(coefs**2, dim=1)), self.gamma))
        # image gradient
        elif isinstance(z, torch.Tensor):
            norm = torch.sum(
                rho(torch.sqrt(torch.sum(z**2, dim=(3, 4))), self.gamma))
        else:
            raise NotImplementedError()
        return norm

    def prox(self, z, tau):
        # wavelet coefficients are given in the form of a tuple of a torch tensor and a list of torch tensors
        if isinstance(z, WaveletCoefficients):
            norms_l = torch.sqrt(torch.sum(z[0]**2, dim=1,
                                           keepdim=True)).expand(*z[0].shape)
            prox_l = torch.zeros_like(z[0])
            ind = norms_l > 0
            prox_l[ind] = prox_rho(norms_l[ind], self.gamma,
                                   tau) * z[0][ind] / norms_l[ind]
            prox_h = []
            for coefs in z[1]:
                norms = torch.sqrt(torch.sum(
                    coefs**2, dim=1, keepdim=True)).expand(*coefs.shape)
                prox = torch.zeros_like(coefs)
                ind = norms > 0.
                prox[ind] = prox_rho(norms[ind], self.gamma,
                                     tau) * coefs[ind] / norms[ind]
                prox_h.append(prox)
            prox = WaveletCoefficients((prox_l, prox_h))
        # image gradient
        elif isinstance(z, torch.Tensor):
            norms = torch.sqrt(torch.sum(z**2, dim=(3, 4),
                                         keepdim=True)).expand(*z.shape)
            prox = torch.zeros_like(z)
            ind = norms > 0.
            prox[ind] = prox_rho(norms[ind], self.gamma,
                                 tau) * z[ind] / norms[ind]
        else:
            raise NotImplementedError()
        return prox

    def grad(self, z):
        # wavelet coefficients are given in the form of a tuple of a torch tensor and a list of torch tensors
        if isinstance(z, WaveletCoefficients):
            norms_l = torch.sqrt(torch.sum(z[0]**2, dim=1,
                                           keepdim=True)).expand(*z[0].shape)
            grad_l = phi(norms_l, self.gamma) * z[0]
            grad_h = []
            for coefs in z[1]:
                norms = torch.sqrt(torch.sum(
                    coefs**2, dim=1, keepdim=True)).expand(*coefs.shape)
                grad_ = phi(norms, self.gamma) * coefs
                grad_h.append(grad_)
            grad = WaveletCoefficients((grad_l, grad_h))
        # image gradient
        elif isinstance(z, torch.Tensor):
            norms = torch.sqrt(torch.sum(z**2, dim=(3, 4),
                                         keepdim=True)).expand(*z.shape)
            grad = phi(norms, self.gamma) * z
        else:
            raise NotImplementedError()
        return grad

    def hess(self, z, w):
        # wavelet coefficients are given in the form of a tuple of a torch tensor and a list of torch tensors
        if isinstance(z, WaveletCoefficients):
            norms_l = torch.sqrt(torch.sum(z[0]**2, dim=1,
                                           keepdim=True)).expand(*z[0].shape)
            hess_l = psi(norms_l, self.gamma) * z[0] * torch.sum(
                z[0] * w[0], dim=1, keepdim=True) + phi(norms_l,
                                                        self.gamma) * w[0]
            hess_h = []
            for z_coefs, w_coefs in zip(z[1], w[1]):
                norms = torch.sqrt(torch.sum(
                    z_coefs**2, dim=1, keepdim=True)).expand(*z_coefs.shape)
                hess_ = psi(norms, self.gamma) * z_coefs * torch.sum(
                    z_coefs * w_coefs, dim=1, keepdim=True) + phi(
                        norms, self.gamma) * w_coefs
                hess_h.append(hess_)
            hess_w = WaveletCoefficients((hess_l, hess_h))
        # image gradient
        elif isinstance(z, torch.Tensor):
            norms = torch.sqrt(torch.sum(z**2, dim=(3, 4),
                                         keepdim=True)).expand(*z.shape)
            hess_w = psi(norms, self.gamma) * z * torch.sum(
                z * w, dim=(3, 4), keepdim=True) + phi(norms, self.gamma) * w
        else:
            raise NotImplementedError()
        return hess_w
