import torch


class CartesianLineParametrisation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param, n1, n2, line_dim=0):
        if line_dim == 0:
            S = param.view(1, n2).expand(n1, n2)
        else:
            S = param.view(n1, 1).expand(n1, n2)
        ctx.line_dim = line_dim
        return S

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.line_dim == 0:
            grad_param = torch.sum(grad_out, dim=0)
        else:
            grad_param = torch.sum(grad_out, dim=1)
        return grad_param, None, None, None


cart_line_parametrisation = CartesianLineParametrisation.apply
