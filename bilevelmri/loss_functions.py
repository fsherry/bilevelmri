from .linear_ops.gradients import Grad
from .functionals import Smoothed1Norm
import torch


def least_squares(x, x_star):
    return torch.mean((x - x_star)**2)


class EdgeAligned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_star, gamma):
        ctx.reg_func = Smoothed1Norm(gamma=gamma)
        ctx.lin_op = Grad()
        diff = ctx.lin_op(x - x_star)
        ctx.save_for_backward(diff, x)
        return ctx.reg_func(ctx.lin_op(x - x_star)) / torch.numel(x)

    @staticmethod
    def backward(ctx, grad_out):
        diff, x = ctx.saved_tensors
        gradx = ctx.lin_op.T(ctx.reg_func.grad(diff)) / torch.numel(x)
        gradx_star = -gradx
        return gradx, gradx_star, None


def edge_aligned(x, x_star, gamma=1e-2):
    return EdgeAligned.apply(x, x_star, gamma)
