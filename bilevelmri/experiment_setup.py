from .lower_level import lower_level_solver
from .objective_function import obj_func_general_parametrisation, ObjectiveTracker

import numpy as np
import torch
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import datetime


def learn(data,
          p_init,
          p_bounds,
          parametrisation,
          A,
          reg_func,
          loss,
          penalty,
          params,
          track=False):
    if track:
        tracker = ObjectiveTracker(
            params, parametrisation, print_to_stdout=True)

        @tracker
        def obj(p, data, params):
            f_data, f_pen, g = obj_func_general_parametrisation(
                p, data, parametrisation, A, reg_func, loss, penalty, params)
            return f_data, f_pen, g
    else:
        counter = 0

        def callback(p):
            nonlocal counter
            counter += 1
            S, alpha, eps = parametrisation(torch.tensor(p), params)
            S = S.reshape(-1).cpu().numpy()
            alpha = alpha.cpu().numpy()
            print(
                '\nIteration #{}: Current sampling rate {:.1f}%, alpha {:.2e}, eps {:.2e}'
                .format(counter,
                        np.mean(S > 0) * 100, alpha.item(), eps.item()))

        def obj(p, data, params):
            f_data, f_pen, g = obj_func_general_parametrisation(
                p, data, parametrisation, A, reg_func, loss, penalty, params)
            return f_data + f_pen, g

    start_time = datetime.datetime.now()
    if 'pgtol' in params['alg_params']['LBFGSB']:
        pgtol = params['alg_params']['LBFGSB']['pgtol']
    else:
        pgtol = 1e-10
    if 'maxit' in params['alg_params']['LBFGSB']:
        maxiter = params['alg_params']['LBFGSB']['maxit']
    else:
        maxiter = 1000
    print('Learning sampling pattern:')
    p, _, info = fmin_l_bfgs_b(
        lambda p: obj(p, data, params),
        p_init,
        bounds=p_bounds,
        pgtol=pgtol,
        factr=0,
        maxiter=maxiter,
        callback=tracker.callback if track else callback)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    results = {'elapsed_time': elapsed_time, 'p': p, 'info': info}
    if track:
        results['tracker'] = tracker
    return results


def compute_statistics(data, p, A, reg_func, parametrisation, params):
    recons = []
    chunks = data['y'].split(10)
    for chunk in chunks:
        S, alpha, eps = parametrisation(
            torch.tensor(p, device=chunk.device), params)
        if 'xinit' in params['alg_params']['ll_sol'] and params['alg_params'][
                'll_sol']['xinit'].shape[0] != chunk.shape[0]:
            del params['alg_params']['ll_sol']['xinit']
        chunk_recon = lower_level_solver(chunk, S, alpha, eps, A, reg_func,
                                         params['alg_params'])
        recons.append(chunk_recon.to('cpu'))
    recons = torch.cat(recons, dim=0)
    ssims = []
    psnrs = []
    for i in range(recons.shape[0]):
        abs_recon = torch.sqrt(torch.sum(recons[i, :, :, :]**2, dim=2)).numpy()
        abs_clean = torch.sqrt(torch.sum(data['x'][i, :, :, :]**2,
                                         dim=2)).cpu().numpy()
        ssims.append(ssim(abs_clean, abs_recon))
        psnrs.append(psnr(abs_clean, abs_recon))
    results = {'recons': recons, 'ssims': ssims, 'psnrs': psnrs}
    return results
