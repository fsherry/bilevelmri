import torch


def grad(x):
    """Compute a finite difference approximation to the gradient for a minibatch of
    complex 2D images with a zero Neumann boundary condition.

    :param x: a torch.Tensor of shape [n, h, w, 2], with n the minibatch size,
    h and w respectively the height and width of the images, and the last
    dimension encoding the real and imaginary components of the images.
    :returns: a torch.Tensor of shape [n, h, w, 2, 2]. The first 4 dimensions
    have the same meaning as they do in x and the last dimension encodes the
    direction in which the finite difference is taken
    :rtype: torch.Tensor

    """

    xshp, xdev, xdt = x.shape, x.device, x.dtype
    grad_x = torch.cat(
        (x[:, 1:, :, :] - x[:, :-1, :, :],
         torch.zeros(xshp[0], 1, xshp[2], 2, device=xdev, dtype=xdt)),
        dim=1)
    grad_y = torch.cat(
        (x[:, :, 1:, :] - x[:, :, :-1, :],
         torch.zeros(xshp[0], xshp[1], 1, 2, device=xdev, dtype=xdt)),
        dim=2)
    return torch.stack((grad_x, grad_y), dim=4)


def div(x):
    """Compute a finite difference approximation to the divergence for a minibatch
    of complex 2D vector field with a zero Neumann boundary condition.

    :param x: a torch.Tensor of shape [n, h, w, 2, 2], with n the minibatch
    size, h and w respectively the height and width of the vector fields, the
    dimension before last encoding the real and imaginary components of the
    vector fields, and the last dimension encoding the components of the vector
    fields in the height and width direction.
    :returns: a torch.Tensor of shape [n, h, w, 2], which represents the
    divergence of the input vector fields. Its dimensions have the same meaning
    as the first 4 dimensions of x.
    :rtype: torch.Tensor

    """

    diff_x = torch.cat(
        (-x[:, 0:1, :, :, 0], x[:, :-2, :, :, 0] - x[:, 1:-1, :, :, 0],
         x[:, -2:-1, :, :, 0]),
        dim=1)
    diff_y = torch.cat(
        (-x[:, :, 0:1, :, 1], x[:, :, :-2, :, 1] - x[:, :, 1:-1, :, 1],
         x[:, :, -2:-1, :, 1]),
        dim=2)
    return -(diff_x + diff_y)
