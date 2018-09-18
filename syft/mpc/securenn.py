from syft.mpc.spdz import (spdz_add, spdz_mul,
                           generate_zero_shares_communication,
                           Q_BITS, field)
import torch


def decompose(tensor):
    powers = torch.arange(Q_BITS)
    for i in range(len(tensor.shape)):
        powers = powers.unsqueeze(0)
    tensor = tensor.unsqueeze(-1)
    moduli = 2 ** powers
    tensor = (tensor / moduli.long()) % 2
    return tensor.short()


def select_shares(alpha, x, y, workers, mod=field):
    """
    alpha is a shared binary tensor
    x and y are private tensors to choose elements or slices from
        (following broadcasting rules)

    all of type _GeneralizedPointerTensor

    Computes z = (1 - alpha) * x + alpha * y
    """
    x_negative = (-1) * x
    w = y + x_negative
    c = alpha * w #spdz_mul(alpha, w, workers, mod=mod)
    u = generate_zero_shares_communication()
    z = spdz_add(spdz_add(x, c), u)
    return z


def private_compare(x, r, beta):
    """
    computes beta XOR (x > r)

    x is private input
    r is public input for comparison
    beta is public random bit tensor

    all of type _GeneralizedPointerTensor
    """
    t = (r + 1) % (2 ** Q_BITS)

    r_bits = decompose(r)
    t_bits = decompose(t)

    zeros = beta == 0
    ones = beta == 1
    others = r == 2 ** Q_BITS - 1
    ones = ones & (others - 1).abs()
    # TODO: everything below here

    raise NotImplementedError


def pc_beta0(*args):
    pass


def pc_beta1(*args):
    pass


def pc_else(*args):
    pass
