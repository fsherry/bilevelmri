import torch

from ._lower_level_hessian import lower_level_hessian, lower_level_mixed_derivs
from ..utils.adjoint_solver import cg
from ._lower_level_solver import solver


class LowerLevelSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, S, alpha, eps, A, reg_func, alg_params):
        recon = solver(y, S, alpha, eps, A, reg_func, **alg_params['ll_sol'])
        ctx.save_for_backward(recon, y, S, alpha, eps)
        ctx.A = A
        ctx.reg_func = reg_func
        ctx.alg_params = alg_params
        return recon

    @staticmethod
    def backward(ctx, grad_out):
        recon, y, S, alpha, eps = ctx.saved_tensors
        A, reg_func, alg_params = ctx.A, ctx.reg_func, ctx.alg_params

        def hess(w):
            return lower_level_hessian(recon, w, S, alpha, eps, A, reg_func)

        lin_sol = cg(hess, grad_out, **alg_params['lin_sys'])
        grady, gradS, gradalpha, gradeps = lower_level_mixed_derivs(
            recon, lin_sol, y, S, alpha, eps, A, reg_func)
        grads = (-grady, -gradS, -gradalpha, -gradeps, None, None, None)
        return grads


lower_level_solver = LowerLevelSolver.apply
