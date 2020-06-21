from .linear_operator import LinearOperator
from .wavelet_coefficients import WaveletCoefficients
import torch

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError:
    print(
        'pytorch_wavelets could not be imported; it is available at https://github.com/fbcotter/pytorch_wavelets'
    )


class Wavelet(LinearOperator):
    def __init__(self, J=3, wave='db4', device='cpu', dtype=torch.float64):
        super().__init__()
        _dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        self.dwt = DWTForward(J=J, wave=wave, mode='per').to(device)
        self.idwt = DWTInverse(wave=wave, mode='per').to(device)
        torch.set_default_dtype(_dtype)
        self.norm_bound = 1.

    def __call__(self, x):
        return WaveletCoefficients(self.dwt(x.permute(0, 3, 1, 2)))

    def T(self, z):
        return self.idwt(z.coefs).permute(0, 2, 3, 1)
