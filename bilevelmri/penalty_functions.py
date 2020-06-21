import torch


def l1_penalty(S, beta=1.):
    return torch.sum(beta * S) / torch.numel(S)


def l1_disc_penalty(S, beta=(1., 1.)):
    return torch.sum(beta[0] * S + beta[1] * S * (1 - S)) / torch.numel(S)


def penalty_lifted(penalty, params):
    def line_penalty(p):
        n1, n2 = params['model']['n1'], params['model']['n2']
        n = n2 if params['model']['line_dim'] == 0 else n1
        return penalty(p[:n])

    return line_penalty
