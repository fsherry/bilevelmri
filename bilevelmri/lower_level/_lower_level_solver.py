from math import sqrt
import torch

from ..linear_ops.wavelet_coefficients import WaveletCoefficients


def solver(y, S, alpha, eps, A, reg_func, xinit=None, maxit=500, tol=1e-10):
    if alpha == 0.:
        print(
            'Lower level solver: alpha equals 0, so we can solve the lower level problem explicitly with the FFT'
        )
        return torch.ifft(
            S.view(1, *S.shape, 1)**2 * y / (S.view(1, *S.shape, 1)**2 + eps),
            signal_ndim=2,
            normalized=True)
    else:

        def K(x):
            return (x, A(x))

        def Kstar(z):
            return z[0] + A.T(z[1])

        def proxF1star(z, S, sigma):
            S = S.view(1, *S.shape, 1)
            inner = (torch.fft(z, 2, normalized=True) + S**2 * y) / (
                sigma + S**2)
            return z - sigma * torch.ifft(inner, 2, normalized=True)

        def proxF2star(z, alpha, sigma):
            return z - sigma * reg_func.prox(z / sigma, alpha / sigma)

        def proxG(x, eps, tau):
            return x / (1 + eps * tau)

        conv_constG = eps.item()
        F2_smoothness = alpha.item() * reg_func.smoothness_bound
        maxS = torch.max(S**2).item()
        if F2_smoothness > 0. and maxS > 0.:
            conv_constFstar = min(1 / maxS, 1 / F2_smoothness)
        elif maxS > 0.:
            conv_constFstar = 1 / maxS
        elif F2_smoothness > 0.:
            conv_constFstar = 1 / F2_smoothness
        else:
            conv_constFstar = 1e5

        op_norm = sqrt(1. + A.norm_bound**2)
        if conv_constG > 0.:
            mu = 2 * sqrt(conv_constFstar * conv_constG) / op_norm
            step_tau = mu / (2 * conv_constG)
            step_sigma = mu / (2 * conv_constFstar)
            theta = 1 / (1 + mu)
        else:
            step_tau = 1 / op_norm
            step_sigma = 1 / op_norm

        interval = 125

        if xinit is None:
            xinit = torch.ifft(
                S.view(1, *S.shape, 1)**2 * y, 2, normalized=True)
        x = xinit
        z = K(x)
        xbar = x
        print(
            'Solver for lower level problem running on minibatch of size {0}'.
            format(x.shape[0]))
        for i in range(1, maxit + 1):
            xold = x
            zold = z
            K_xbar = K(xbar)
            z_0 = proxF1star(z[0] + step_sigma * K_xbar[0], S, step_sigma)
            z_1 = proxF2star(z[1] + step_sigma * K_xbar[1], alpha, step_sigma)
            z = (z_0, z_1)
            x = proxG(x - step_tau * Kstar(z), eps, step_tau)
            if not conv_constG > 0.:
                theta = 1. / sqrt(1. + 2. * conv_constFstar * step_sigma)
                step_sigma = theta * step_sigma
                step_tau = theta * step_tau
            xbar = x + theta * (x - xold)

            rel_err_x = torch.sqrt(
                torch.sum((x - xold)**2) / (1e-6 + torch.sum(xold**2)))

            err_sqr = torch.sum((z[0] - zold[0])**2)
            old_norm_sqr = torch.sum(zold[0]**2)
            diff_z = z[1] - zold[1]
            if isinstance(z[1], WaveletCoefficients):
                err_sqr += torch.sum(diff_z.coefs[0]**2)
                old_norm_sqr += torch.sum(zold[1].coefs[0]**2)
                for diff_coefs, zold_coefs in zip(diff_z.coefs[1],
                                                  zold[1].coefs[1]):
                    err_sqr += torch.sum(diff_coefs**2)
                    old_norm_sqr += torch.sum(zold_coefs**2)
            else:
                err_sqr += torch.sum(diff_z**2)
                old_norm_sqr += torch.sum(zold[1]**2)
            rel_err_z = torch.sqrt(err_sqr / (1e-6 + old_norm_sqr))
            rel_err = rel_err_x + rel_err_z
            if rel_err < tol:
                print('Finishing at iteration {0}: Relative error: {1:.2e}'.
                      format(i, rel_err.item()))
                break
            if i % interval == 0:
                print('Iteration {0}: Relative error: {1:.2e}'.format(
                    i, rel_err.item()))
        return x


def lower_level_objective(x, y, S, alpha, eps, A, reg_func):
    return 0.5 * torch.sum(
        (S.view(1, *S.shape, 1) * torch.fft(x, signal_ndim=2, normalized=True)
         - y)**2) + alpha * reg_func(A(x)) + 0.5 * eps * torch.sum(x**2)


def lower_level_objective_grad(x, y, S, alpha, eps, A, reg_func):
    data_grad = torch.ifft(
        S.view(1, *S.shape, 1)**2 *
        (torch.fft(x, signal_ndim=2, normalized=True) - y),
        signal_ndim=2,
        normalized=True)
    return data_grad + alpha * A.T(reg_func.grad(A(x))) + eps * x
