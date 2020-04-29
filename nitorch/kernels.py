# -*- coding: utf-8 -*-
"""Convolution kernels.

@author: yael.balbastre@gmail.com
"""

# TODO:
# . Implement Sinc kernel
# . Use inplace operations if gradients not required

# WARNING:
# . Currently, fwhm and voxel sizes are ordered as [depth wifth height]
#   I am not sure yet what convention is best

import torch
from nitorch import utils

__all__ = ['smooth', 'energy', 'energy1d', 'energy2d', 'energy3d',
           'make_separable', 'imgrad']


def make_separable(ker, channels):
    """Transform a single-channel kernel into a multi-channel separable kernel.

    Args:
        ker (torch.tensor): Single-channel kernel (1, 1, D, H, W).
        channels (int): Number of input/output channels.

    Returns:
        ker (torch.tensor): Multi-channel group kernel (1, 1, D, H, W).

    """
    ndim = torch.as_tensor(ker.shape).numel()
    repetitions = (channels,) + (1,)*(ndim-1)
    ker = ker.repeat(repetitions)
    return ker


def integrate_poly(l, h, *args):
    """Integrate a polynomial on an interval.

    k = integrate_poly(l, h, a, b, c, ...)
    integrates the polynomial a+b*x+c*x^2+... on [l,h]

    All inputs should be `torch.Tensor`
    """
    # NOTE: operations are not performed inplace (+=, *=) so that autograd
    # can backpropagate.
    # TODO: (maybe) use inplace if gradients not required
    zero = torch.zeros(tuple(), dtype=torch.bool)
    k = torch.zeros(l.shape, dtype=l.dtype, device=l.device)
    hh = h
    ll = l
    for i in range(len(args)):
        if torch.any(args[i] != zero):
            k = k + (args[i]/(i+1))*(hh-ll)
        hh = hh * h
        ll = ll * l
    return(k)


def gauss1d(fwhm, basis, x):
    if basis:
        return(gauss1d1(fwhm, x))
    else:
        return(gauss1d0(fwhm, x))


def rect1d(fwhm, basis, x):
    if basis:
        return(rect1d1(fwhm, x))
    else:
        return(rect1d0(fwhm, x))


def triangle1d(fwhm, basis, x):
    if basis:
        return(triangle1d1(fwhm, x))
    else:
        return(triangle1d0(fwhm, x))


def gauss1d0(w, x):
    logtwo = torch.tensor(2., dtype=w.dtype, device=w.device).log()
    sqrttwo = torch.tensor(2., dtype=w.dtype, device=w.device).sqrt()
    s = w/(8.*logtwo).sqrt() + 1E-7  # standard deviation
    if x is None:
        lim = torch.floor(4*s+0.5).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    w1 = 1./(sqrttwo*s)
    ker = 0.5*((w1*(x+0.5)).erf() - (w1*(x-0.5)).erf())
    ker = ker.clamp(min=0)
    return ker, x


def gauss1d1(w, x):
    import math
    logtwo = torch.tensor(2., dtype=w.dtype, device=w.device).log()
    sqrttwo = torch.tensor(2., dtype=w.dtype, device=w.device).sqrt()
    sqrtpi = torch.tensor(math.pi, dtype=w.dtype, device=w.device).sqrt()
    s = w/(8.*logtwo).sqrt() + 1E-7  # standard deviation
    if x is None:
        lim = torch.floor(4*s+1).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    w1 = 0.5*sqrttwo/s
    w2 = -0.5/s.pow(2)
    w3 = s/(sqrttwo*sqrtpi)
    ker = 0.5*((w1*(x+1)).erf()*(x+1)
               + (w1*(x-1)).erf()*(x-1)
               - 2*(w1*x).erf()*x) \
        + w3*((w2*(x+1).pow(2)).exp()
              + (w2*(x-1).pow(2)).exp()
              - 2*(w2*x.pow(2)).exp())
    ker = ker.clamp(min=0)
    return ker, x


def rect1d0(w, x):
    if x is None:
        lim = torch.floor((w+1)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=w.dtype, device=w.device)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    ker = torch.max(torch.min(x+0.5, w/2) - torch.max(x-0.5, -w/2), zero)
    ker = ker/w
    return ker, x


def rect1d1(w, x):
    if x is None:
        lim = torch.floor((w+2)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=torch.float)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_low = torch.min(torch.max(x-w/2, -one),   zero)
    neg_upp = torch.max(torch.min(x+w/2,  zero), -one)
    pos_low = torch.min(torch.max(x-w/2,  zero),  one)
    pos_upp = torch.max(torch.min(x+w/2,  one),   zero)
    ker = integrate_poly(neg_low, neg_upp, one,  one) \
        + integrate_poly(pos_low, pos_upp, one, -one)
    ker = ker/w
    return ker, x


def triangle1d0(w, x):
    if x is None:
        lim = torch.floor((2*w+1)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=torch.float)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_low = torch.min(torch.max(x-0.5, -w),     zero)
    neg_upp = torch.max(torch.min(x+0.5,  zero), -w)
    pos_low = torch.min(torch.max(x-0.5,  zero),  w)
    pos_upp = torch.max(torch.min(x+0.5,  w),     zero)
    ker = integrate_poly(neg_low, neg_upp, one,  1/w) \
        + integrate_poly(pos_low, pos_upp, one, -1/w)
    ker = ker/w
    return ker, x


def triangle1d1(w, x):
    if x is None:
        lim = torch.floor((2*w+2)/2).type(torch.int)
        x = torch.tensor(range(-lim, lim+1), dtype=torch.float)
    zero = torch.zeros(tuple(), dtype=w.dtype, device=w.device)
    one = torch.ones(tuple(), dtype=w.dtype, device=w.device)
    neg_neg_low = torch.min(torch.max(x,   -one),   zero)
    neg_neg_upp = torch.max(torch.min(x+w,  zero), -one)
    neg_pos_low = torch.min(torch.max(x,    zero),  one)
    neg_pos_upp = torch.max(torch.min(x+w,  one),   zero)
    pos_neg_low = torch.min(torch.max(x-w, -one),   zero)
    pos_neg_upp = torch.max(torch.min(x,    zero), -one)
    pos_pos_low = torch.min(torch.max(x-w,  zero),  one)
    pos_pos_upp = torch.max(torch.min(x,    one),   zero)
    ker = integrate_poly(neg_neg_low, neg_neg_upp, 1+x/w,  1+x/w-1/w, -1/w) \
        + integrate_poly(neg_pos_low, neg_pos_upp, 1+x/w, -1-x/w-1/w,  1/w) \
        + integrate_poly(pos_neg_low, pos_neg_upp, 1-x/w,  1-x/w+1/w,  1/w) \
        + integrate_poly(pos_pos_low, pos_pos_upp, 1-x/w, -1+x/w+1/w, -1/w)
    ker = ker/w
    return ker, x


smooth_switcher = {
    'gauss': gauss1d,
    'rect': rect1d,
    'triangle': triangle1d,
    0: rect1d,
    1: triangle1d,
    }


def smooth(type, fwhm=1, basis=0, x=None, sep=True):
    """Create a smoothing kernel.

    Creates a (separable) smoothing kernel with fixed (i.e., not learned)
    weights. These weights are obtained by analytically convolving a
    smoothing function (e.g., Gaussian) with a basis function that encodes
    the underlying image (e.g., trilinear).
    Note that `smooth` is fully differentiable with respect to `fwhm`.
    If the kernel is evaluated at all integer coordinates from its support,
    its elements are ensured to sum to one.
    The returned kernel is a `torch.Tensor`.

    Args:
        type (str,int): Smoothing function (integrates to one).
            . 0, 'rect': Rectangular function (0th order B-spline)
            . 1, 'tri': Triangular function (1st order B-spline)
            . 'gauss': Gaussian
            . 'sinc': Sinc
        fwhm (array_like,float,optional): Full-width at half-maximum of the
            smoothing function (in voxels), in each dimension.
            Default: 1.
        basis (array_like,int,optional): Image encoding basis (B-spline order)
            Default: 0
        x (tuple,array_like,optional): Coordinates at which to evaluate the
            kernel. If None, evaluate at all integer coordinates from its
            support (truncated support for 'gauss' and 'sinc' kernels).
            Default: None
        sep(boolean): Return separable 1D kernels. If False, the 1D kernels
            are combined to form an N-D kernel.
            Default: True

    Returns:
        If `sep is False` or all input parameters are scalar: a `torch.Tensor`
        Else: a tuple of `torch.Tensor`


    """
    # Convert to tensors
    fwhm = torch.as_tensor(fwhm)
    if not fwhm.is_floating_point():
        fwhm = fwhm.type(torch.float)
    basis = torch.as_tensor(basis)
    return_tuple = True
    if not isinstance(x, tuple):
        return_tuple = not (fwhm.shape == torch.Size([]) and
                            basis.shape == torch.Size([]))
        x = (x,)
    x = tuple(torch.as_tensor(x1).flatten() if x1 is not None else None
              for x1 in x)

    # Ensure all sizes are consistant
    fwhm = fwhm.flatten()
    basis = basis.flatten()
    nker = max(fwhm.numel(), basis.numel(), len(x))
    fwhm = torch.cat((fwhm, fwhm[-1].repeat(max(0, nker-fwhm.numel()))))
    basis = torch.cat((basis, basis[-1].repeat(max(0, nker-basis.numel()))))
    x = x + (x[-1],)*max(0, nker-len(x))

    # Loop over dimensions
    ker = tuple()
    x = list(x)
    for d in range(nker):
        ker1, x[d] = smooth_switcher[type](fwhm[d], basis[d], x[d])
        shape = [1, ] * nker
        shape[d] = ker1.numel()
        ker1 = ker1.reshape(shape)
        ker1 = ker1.unsqueeze(0).unsqueeze(0)  # Cout = 1, Cin = 1
        ker += (ker1, )

    # Make N-D kernel
    if not sep:
        ker1 = ker
        ker = ker1[0]
        for d in range(1, nker):
            ker = ker * ker1[d]
    elif not return_tuple:
        ker = ker[0]

    return ker


def energy(dim, absolute=0, membrane=0, bending=0, lame=(0, 0), vs=1,
           displacement=False, dtype=None, device=None):
    """Generate a convolution kernel for a mixture of differential energies.

    This function builds a convolution kernel that embeds a mixture of
    differential energies. In practice, this energy can be computed as
    E = <f,k*f>, where <.,.> is the Eucldean dot product.
    Possible energies are:
        . absolute = sum of squared absolute values
        . membrane = sum of squared first derivatives
        . bending  = sum of squared second derivatives (diagonal terms only)
        . lame     = linear elastic energy
            [0] sum of divergences  (Lame's 1st parameter, lambda)
            [1] sum of shears       (Lame's 2nd parameter, mu)

    Note: The lame parameters should be entered in the opposite order from SPM
          SPM: (mu, lambda) / nitorch: (lambda, mu)

    Args:
        dim (int): Dimension of the problem (1, 2, or 3)
        absolute (float, optional): Defaults to 0.
        membrane (float, optional): Defaults to 0.
        bending (float, optional): Defaults to 0.
        lame (float, optional): Defaults to (0, 0).
        vs (float, optional): Defaults to 1.
        displacement (bool, optional): True if input field is a displacement
            field. Defaults to True if `linearelastic != (0, 0)`, else False.
        dtype (torch.dtype, optional)
        device (torch.device, optional)

    Raises:
        ValueError: DESCRIPTION.

    Returns:
        ker (TYPE): DESCRIPTION.

    """
    # Check arguments
    absolute = torch.as_tensor(absolute)
    if absolute.numel() != 1:
        raise ValueError('The absolute energy must be parameterised by '
                         'exactly one parameter. Received {}.'
                         .format(absolute.numel()))
    absolute = absolute.flatten()[0]
    membrane = torch.as_tensor(membrane)
    if membrane.numel() != 1:
        raise ValueError('The membrane energy must be parameterised by '
                         'exactly one parameter. Received {}.'
                         .format(membrane.numel()))
    membrane = membrane.flatten()[0]
    bending = torch.as_tensor(bending)
    if bending.numel() != 1:
        raise ValueError('The bending energy must be parameterised by '
                         'exactly one parameter. Received {}.'
                         .format(bending.numel()))
    bending = bending.flatten()[0]
    lame = torch.as_tensor(lame)
    if lame.numel() != 2:
        raise ValueError('The linear-elastic energy must be parameterised by '
                         'exactly two parameters. Received {}.'
                         .format(lame.numel()))
    lame = lame.flatten()

    if (lame != 0).any():
        displacement = True

    # Kernel size
    if bending != 0:
        kdim = 5
    elif membrane != 0 or (lame != 0).any():
        kdim = 3
    elif absolute != 0:
        kdim = 1
    else:
        kdim = 0

    # Compute 1/vs^2
    vs = torch.as_tensor(vs).flatten()
    vs = torch.cat((vs, vs[-1].repeat(max(0, dim-vs.numel()))))
    vs = 1./(vs*vs)

    # Accumulate energies
    ker = torch.zeros((1, 1) + (kdim,)*dim, dtype=dtype, device=device)
    if absolute != 0:
        kpad = ((kdim-1)/2,)*dim
        ker1 = energy_absolute(dim, vs, dtype, device)
        ker1 = utils.pad(ker1, kpad, side='both')
        if displacement:
            ker2 = ker1
            ker1 = torch.zeros((dim, dim) + (kdim,)*dim,
                               dtype=dtype, device=device)
            for d in range(dim):
                ker1[d, d, ...] = ker2/vs[d]
        ker = ker + absolute*ker1

    if membrane != 0:
        kpad = ((kdim-3)/2,)*dim
        ker1 = energy_membrane(dim, vs, dtype, device)
        ker1 = utils.pad(ker1, kpad, side='both')
        if displacement:
            ker2 = ker1
            ker1 = torch.zeros((dim, dim) + (kdim,)*dim,
                               dtype=dtype, device=device)
            for d in range(dim):
                ker1[d, d, ...] = ker2/vs[d]
        ker = ker + membrane*ker1

    if bending != 0:
        ker1 = energy_bending(dim, vs, dtype, device)
        if displacement:
            ker2 = ker1
            ker1 = torch.zeros((dim, dim) + (kdim,)*dim,
                               dtype=dtype, device=device)
            for d in range(dim):
                ker1[d, d, ...] = ker2/vs[d]
        ker = ker + bending*ker1

    if lame[0] != 0:
        kpad = ((kdim-3)/2,)*dim
        ker1 = energy_linearelastic(dim, vs, 1, dtype, device)
        ker1 = utils.pad(ker1, kpad, side='both')
        ker = ker + lame[0]*ker1

    if lame[1] != 0:
        kpad = ((kdim-3)/2,)*dim
        ker1 = energy_linearelastic(dim, vs, 2, dtype, device)
        ker1 = utils.pad(ker1, kpad, side='both')
        ker = ker + lame[1]*ker1

    return ker


def energy1d(*args, **kwargs):
    """energy(1, absolute, membrane, bending, linearelastic, vs)."""
    return(energy(1, *args, **kwargs))


def energy2d(*args, **kwargs):
    """energy(2, absolute, membrane, bending, linearelastic, vs)."""
    return(energy(2, *args, **kwargs))


def energy3d(*args, **kwargs):
    """energy(3, absolute, membrane, bending, linearelastic, vs)."""
    return(energy(3, *args, **kwargs))


def energy_absolute(dim, vs, dtype=torch.float, device='cpu'):
    ker = torch.ones(1, dtype=dtype, device=device)
    ker = ker.reshape((1,)*dim)
    ker = ker.unsqueeze(0).unsqueeze(0)
    return ker


def energy_membrane(dim, vs, dtype=torch.float, device='cpu'):
    ker = torch.zeros((3,)*dim, dtype=dtype, device=device)
    ker[(1,)*dim] = 2.*vs.sum()
    for d in range(dim):
        ker[tuple(0 if i == d else 1 for i in range(dim))] = -vs[d]
        ker[tuple(2 if i == d else 1 for i in range(dim))] = -vs[d]
    ker = ker.unsqueeze(0).unsqueeze(0)
    return ker


def energy_bending(dim, vs, dtype=torch.float, device='cpu'):
    centre = 6.*vs.pow(2).sum()
    for d in range(dim):
        for dd in range(d+1, dim):
            centre = centre + 8.*vs[d]*vs[dd]
    vs = vs.reshape(vs.numel(), 1)
    ker = torch.zeros((5,)*dim, dtype=dtype, device=device)
    ker[(2,)*dim] = centre
    for d in range(dim):
        ker[tuple(1 if i == d else 2 for i in range(dim))] = -4.*vs[d]*vs.sum()
        ker[tuple(3 if i == d else 2 for i in range(dim))] = -4.*vs[d]*vs.sum()
        ker[tuple(0 if i == d else 2 for i in range(dim))] = vs[d]*vs[d]
        ker[tuple(4 if i == d else 2 for i in range(dim))] = vs[d]*vs[d]
        for dd in range(d+1, dim):
            ker[tuple(1 if i in (d, dd) else 2
                      for i in range(dim))] = 2*vs[d]*vs[dd]
            ker[tuple(3 if i in (d, dd) else 2
                      for i in range(dim))] = 2*vs[d]*vs[dd]
            ker[tuple(1 if i == d else 3 if i == dd else 2
                      for i in range(dim))] = 2*vs[d]*vs[dd]
            ker[tuple(3 if i == d else 1 if i == dd else 2
                      for i in range(dim))] = 2*vs[d]*vs[dd]
    ker = ker.unsqueeze(0).unsqueeze(0)
    return ker


def energy_linearelastic(dim, vs, lame, dtype=torch.float, device='cpu'):
    ker = torch.zeros((dim, dim) + (3,)*dim, dtype=dtype, device=device)
    for d in range(dim):
        ker[(d, d) + (1,)*dim] = 2.
        if lame == 2:
            ker[(d, d) + (1,)*dim] += 2.*vs.sum()/vs[d]
            for dd in range(dim):
                ker[(d, d) + tuple(0 if i == dd else 1 for i in range(dim))] \
                    = -vs[dd]/vs[d] if d != dd else -2.
                ker[(d, d) + tuple(2 if i == dd else 1 for i in range(dim))] \
                    = -vs[dd]/vs[d] if d != dd else -2.
        else:
            ker[(d, d) + tuple(0 if i == d else 1 for i in range(dim))] \
                = -1.
            ker[(d, d) + tuple(2 if i == d else 1 for i in range(dim))] \
                = -1.
        for dd in range(d+1, dim):
            ker[(d, dd) + tuple(0 if i in (d, dd) else 1
                                for i in range(dim))] = -0.25
            ker[(d, dd) + tuple(2 if i in (d, dd) else 1
                                for i in range(dim))] = -0.25
            ker[(d, dd) + tuple(0 if i == d else 2 if i == dd else 1
                                for i in range(dim))] = 0.25
            ker[(d, dd) + tuple(2 if i == d else 0 if i == dd else 1
                                for i in range(dim))] = 0.25
            ker[dd, d, ...] = ker[d, dd, ...]
    return ker


def imgrad(dim, vs=1, which='central', dtype=None, device=None):
    """Kernel that computes the first order gradients of a tensor.

    Args:
        dim (int): Dimension.
        vs (float, optional): Voxel size. Defaults to 1.
        which (tuple, string, optional): Gradient types (one or more):
            . 'forward': forward gradients (next - centre)
            . 'backward': backward gradients (centre - previous)
            . 'central': central gradients ((next - previous)/2)
            Defaults to 'central'.
        dtype (torch.dtype, optional): Data type. Defaults to None.
        device (torch.device, optional): Device. Defaults to None.

    Returns:
        ker (TYPE): DESCRIPTION.

    """
    vs = torch.as_tensor(vs).flatten()
    if not vs.is_floating_point():
        vs = vs.to(torch.float)
    vs = torch.cat((vs, vs[-1].repeat(max(0, dim-vs.numel()))))
    if not isinstance(which, tuple):
        which = (which,)
    coord = tuple((0, 2) if w == 'central' else
                  (1, 2) if w == 'forward' else
                  (0, 1) for w in which)
    ker = torch.zeros((dim, len(which), 1) + (3,) * dim,
                      dtype=dtype, device=device)
    for d in range(dim):
        for i in range(len(which)):
            sub = tuple(coord[i][0] if dd == d else 1 for dd in range(dim))
            ker[(d, i, 0) + sub] = -1./(vs[d]*(coord[i][1]-coord[i][0]))
            sub = tuple(coord[i][1] if dd == d else 1 for dd in range(dim))
            ker[(d, i, 0) + sub] = 1./(vs[d]*(coord[i][1]-coord[i][0]))
    ker = ker.reshape((dim*len(which), 1) + (3,)*dim)
    return ker