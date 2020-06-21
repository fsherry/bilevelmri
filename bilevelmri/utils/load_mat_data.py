from scipy.io import loadmat
import torch


def load_and_reconfigure_mat_data(filename, im_size=(192, 192), device='cpu'):
    data = loadmat(filename)
    n1, n2 = im_size
    x = data['data'][0][0][0]
    y = data['data'][0][0][1]
    x = torch.stack((torch.tensor(x.real), torch.tensor(x.imag)), dim=2).view(
        n1, n2, -1, 2).permute(2, 1, 0, 3)
    y = torch.stack((torch.tensor(y.real), torch.tensor(y.imag)), dim=2).view(
        n1, n2, -1, 2).permute(2, 1, 0, 3)
    if device != 'cpu':
        x = x.to(device)
        y = y.to(device)
    data = {'x': x, 'y': y}
    return data













