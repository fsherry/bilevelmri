import torch


def cg(A, rhs, xinit=None, tol=1e-6, maxit=1000):
    # A pytorch implementation of the conjugate gradient algorithm for solving symmetric positive definite linear systems.
    if xinit is None:
        xinit = torch.zeros_like(rhs)
    rhs_norm = torch.sqrt(torch.sum(rhs**2))
    x = xinit
    r = rhs - A(x)
    p = r
    r_norm = torch.sqrt(torch.sum(r**2))
    rel_err = r_norm / (1e-6 + rhs_norm)
    if rel_err < tol:
        print(
            'Initial guess solved equation within tolerance {0:.2e}: relative error was {1:.2e}'
            .format(tol, rel_err))
        return xinit
    for i in range(1, maxit + 1):
        alpha = torch.sum(r**2) / torch.sum(p * A(p))
        x = x + alpha * p
        rnew = r - alpha * A(p)
        rnew_norm = torch.sqrt(torch.sum(rnew**2))
        rel_err = rnew_norm / (1e-6 + rhs_norm)
        if rel_err < tol:
            print(
                'CG converged within tolerance {0:.2e} at iteration {1}: relative error was {2:.2e}'
                .format(tol, i, rel_err))
            return x
        beta = (rnew_norm / r_norm)**2
        p = rnew + beta * p
        r = rnew
        r_norm = rnew_norm
    print(
        'CG finished after {0} iterations: relative error was {1:.2e}'.format(
            i, rel_err))
    return x
