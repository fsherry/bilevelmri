import torch


class Functional:
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError()

    def prox(self, x, tau):
        raise NotImplementedError()

    def grad(self, x):
        raise NotImplementedError()

    def hess(self, x, w):
        raise NotImplementedError()


class Squared2Norm(Functional):
    def __init__(self):
        super().__init__()
        self.smoothness_bound = 1.

    def __call__(self, x):
        return 0.5 * torch.sum(x**2)

    def prox(self, x, tau):
        return x / (tau + 1.)

    def grad(self, x):
        return x

    def hess(self, x, w):
        return w
