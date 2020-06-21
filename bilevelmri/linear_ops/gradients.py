from .linear_operator import LinearOperator
from ..utils.finite_diff import grad, div
from math import sqrt
import torch


class Grad(LinearOperator):
    def __init__(self):
        super().__init__()
        self.norm_bound = sqrt(8.)

    def __call__(self, x):
        return grad(x)

    def T(self, z):
        return -div(z)


class ProjectedGrad(LinearOperator):
    def __init__(self, vector_field):
        super().__init__()
        self.norm_bound = sqrt(8.)
        self.vector_field = vector_field

    def __call__(self, x):
        grad_x = grad(x)
        return grad_x - torch.sum(
            self.vector_field * grad_x, dim=(3, 4),
            keepdim=True) * self.vector_field

    def T(self, z):
        return -div(
            z - torch.sum(self.vector_field * z, dim=(3, 4), keepdim=True) *
            self.vector_field)


class DTV(LinearOperator):
    def __init__(self, ref_ims, eta=1e-2):
        super().__init__()
        self.norm_bound = sqrt(8.)
        grad_refs = grad(ref_ims)
        self.normalised_refs = grad_refs / torch.sqrt(
            torch.sum(eta**2 + grad_refs**2, dim=(3, 4), keepdim=True))
        self.proj_grad = ProjectedGrad(self.normalised_refs)

    def __call__(self, x):
        return self.proj_grad(x)

    def T(self, z):
        return self.proj_grad.T(z)
