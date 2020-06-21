import torch

from .line_patterns import cart_line_parametrisation


def free_parametrisation(p, params):
    S = p[:-2].view(params['model']['n1'], params['model']['n2'])
    alpha = p[-2]
    eps = p[-1]
    return (S, alpha, eps)


def line_parametrisation(p, params):
    if 'line_dim' in params['model']:
        line_dim = params['model']['line_dim']
    else:
        line_dim = 0
    S = cart_line_parametrisation(p[:-2], params['model']['n1'],
                                  params['model']['n2'], line_dim)
    alpha = p[-2]
    eps = p[-1]
    return (S, alpha, eps)


def lifted_parametrisation(p, params):
    if 'line_dim' in params['model']:
        line_dim = params['model']['line_dim']
    else:
        line_dim = 0
    n1, n2 = params['model']['n1'], params['model']['n2']
    n = n2 if line_dim == 0 else n1
    line_mask = cart_line_parametrisation(p[:n], n1, n2, params)
    free_mask = p[n:n + n1 * n2].view(n1, n2)
    S = line_mask * free_mask
    alpha = p[-2]
    eps = p[-1]
    return (S, alpha, eps)


def alpha_parametrisation(p, params, S=None, eps=1e-2):
    if S is None:
        S = torch.ones(
            params['model']['n1'],
            params['model']['n2'],
            device=p.device,
            dtype=p.dtype)
    eps = torch.tensor(eps, device=p.device, dtype=p.dtype)
    return (S, p[0], eps)
