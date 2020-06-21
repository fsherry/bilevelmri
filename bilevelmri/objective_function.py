import numpy as np
import os
import pickle
import torch

from .lower_level import lower_level_solver


class ObjectiveTracker:
    def __init__(self, params, parametrisation, print_to_stdout=True):
        # save the parametrisation so that the callback can display information not just about the learnable parameters, but the sampling pattern and regularisation parameter
        self.parametrisation = parametrisation
        self.params = params
        self._print = print_to_stdout
        self._ll_counter = 0
        self._outer_counter = 0
        self._next_it = False
        if 'checkpoint' in params['results'] and os.path.exists(
                os.path.join(params['results']['folder'], 'checkpoints/')):
            self._checkpoint = params['results']['checkpoint']
            self._experiment_name = params['results']['experiment_name']
            self._checkpoint_path = os.path.join(
                params['results']['folder'], 'checkpoints/',
                '{}_checkpoint.pkl'.format(self._experiment_name))
        else:
            self._checkpoint = None
        self.f_data = np.array([])
        self.f_pen = np.array([])

    def __call__(self, obj):
        def wrapped_obj(p, data, params):
            self.params = params
            # if warm start is saved and we compute the objective function value for new data with a different number of training pairs, delete the warm start
            if 'xinit' in params['alg_params']['ll_sol']:
                if params['alg_params']['ll_sol']['xinit'].shape[0] != data[
                        'x'].shape[0]:
                    del params['alg_params']['ll_sol']['xinit']
            self._next_it = False if self._next_it else True
            self._ll_counter += 1
            f_data, f_pen, g = obj(p, data, params)
            self._f_data_curr, self._f_pen_curr = f_data, f_pen
            f = f_data + f_pen
            if self._print:
                S, alpha, eps = self.parametrisation(torch.tensor(p), params)
                print(
                    '\nLower level solve #{0}: alpha {1:.2e}, eps {2:.2e}, sampling rate {3:.1f}%\n objective function value:{4:.2e} = data error:{5:.2e} + penalty:{6:.2e}\n'
                    .format(self._ll_counter, alpha.item(), eps.item(),
                            torch.mean(1. * (S.reshape(-1) > 0.)).item() * 100,
                            f, f_data, f_pen))
            return f, g

        wrapped_obj.__name__ = obj.__name__
        wrapped_obj.__qualname__ = obj.__qualname__
        return wrapped_obj

    def callback(self, p):
        self._next_it = True
        self._outer_counter += 1
        self.f_data = np.append(self.f_data, self._f_data_curr)
        self.f_pen = np.append(self.f_pen, self._f_pen_curr)
        if self._print:
            S, alpha, eps = self.parametrisation(torch.tensor(p), self.params)
            S = S.reshape(-1).cpu().numpy()
            alpha = alpha.cpu().numpy()
            print(
                '\nIteration #{}: Current sampling rate {:.1f}%, alpha {:.2e}, eps {:.2e}'
                .format(self._outer_counter,
                        np.mean(S > 0) * 100, alpha.item(), eps.item()))
        if self._checkpoint is not None and self._outer_counter % self._checkpoint == 0:
            # avoid saving warm start reconstruction in checkpoint
            _params = self.params
            if 'xinit' in self.params['alg_params']['ll_sol']:
                del self.params['alg_params']['ll_sol']['xinit']
            with open(self._checkpoint_path, 'wb') as cp_file:
                pickle.dump({'p': p, 'tracker': self}, cp_file)
            self.params = _params


def obj_func_general_parametrisation(p, data, parametrisation, A, reg_func,
                                     loss, penalty, params):
    """
    This function does the computations that need to be done to compute the upper level objective function value and its gradient.
    :param p: a numpy array (or torch tensor) representing the parameters that are being learned.
    :param data: a dictionary with keys 'x' and 'y', representing ground truth and measurements respectively. data['x'] and data['y'] should be torch tensors of size (N, n1, n2, 2), where N is the number of training pairs considered, n1, n2 are the size of the images and the last dimension is used to encode real and complex parts.
    :param parametrisation: a function which takes the parameters to be learned and a dictionary params
    :param loss: a loss function, which takes a torch tensor sols of the same shape as data['x'], the lower level solution, and compares it to the ground truth data['x']
    :param penalty: a penalty function, which takes a sampling pattern in the form of a torch tensor of shape (n1, n2) (with (n1, n2) as in data) and computes a penalty that when traded off with the loss function encourages an optimal balance between reconstruction quality and sparsity of the sampling pattern.
    :param params: a dictionary containing the options for the experiment to be run. In particular it should contain the model parameters in a subdictionary params['model'], and algorithm parameters for the lower level solver and adjoint equation solve in subdictionaries params['alg_params']['ll_sol'] and params['alg_params']['lin_sys'] respectively.
    :returns: f_data, the value of the loss function, f_pen, the value of the penalty function, g, the gradient of the objective function (which has value f_data + f_pen) with respect to the learnable parameters.
    :rtype: tuple, of (float, float, torch.Tensor)

    """
    x = data['x']
    y = data['y']
    n1, n2 = x.shape[1:3]
    p = torch.tensor(p, device=x.device)    
    p.requires_grad = True
    if not p.grad is None:
        p.grad.zero_()
    S, alpha, eps = parametrisation(p, params)
    alg_params = params['alg_params']
    sols = lower_level_solver(y, S, alpha, eps, A, reg_func, alg_params)
    # set lower level initialisation for a warm start for the next call
    params['alg_params']['ll_sol']['xinit'] = sols
    f_data = loss(sols, x)
    f_pen = penalty(p)
    f = f_data + f_pen
    f.backward()
    g = p.grad.detach().cpu().numpy()
    return f_data.item(), f_pen.item(), g
