# -*- coding: utf-8 -*-
"""Spatial deformations (i.e., grids)."""

import torch
from nitorch.core import utils, linalg
from nitorch.core.utils import expand, make_vector
from nitorch.core.py import make_list, prod
from nitorch._C.spatial import BoundType, InterpolationType
from nitorch._C.grid import GridPull, GridPush, GridCount, GridGrad
from ._affine import affine_resize, affine_lmdiv
from ._regularisers import solve_grid_sym
from ._finite_differences import diff


__all__ = ['grid_pull', 'grid_push', 'grid_count', 'grid_grad', 'grid_inv',
           'identity_grid', 'affine_grid', 'resize', 'resize_grid', 'reslice',
           'grid_jacobian', 'grid_jacdet',
           'BoundType', 'InterpolationType']

_doc_interpolation = \
"""`interpolation` can be an int, a string or an InterpolationType.
    Possible values are:
        - 0 or 'nearest'    or InterpolationType.nearest
        - 1 or 'linear'     or InterpolationType.linear
        - 2 or 'quadratic'  or InterpolationType.quadratic
        - 3 or 'cubic'      or InterpolationType.cubic
        - 4 or 'fourth'     or InterpolationType.fourth
        - etc.
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific interpolation orders."""

_doc_bound = \
"""`bound` can be an int, a string or a BoundType.
    Possible values are:
        - 'replicate'  or BoundType.replicate
        - 'dct1'       or BoundType.dct1
        - 'dct2'       or BoundType.dct2
        - 'dst1'       or BoundType.dst1
        - 'dst2'       or BoundType.dst2
        - 'dft'        or BoundType.dft
        - 'zero'       or BoundType.zero
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific boundary conditions.
    Note that
    - `dft` corresponds to circular padding
    - `dct2` corresponds to Neumann boundary conditions (symmetric)
    - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
    See https://en.wikipedia.org/wiki/Discrete_cosine_transform
        https://en.wikipedia.org/wiki/Discrete_sine_transform
    """


def grid_pull(input, grid, interpolation='linear', bound='zero', extrapolate=False):
    """Sample an image with respect to a deformation field.

    Notes
    -----
    {interpolation}

    {bound}
    
    If the input dtype is not a floating point type, the input image is 
    assumed to contain labels. Then, unique labels are extracted 
    and resampled individually, making them soft labels. Finally, 
    the label map is reconstructed from the individual soft labels by 
    assigning the label with maximum soft value.

    Parameters
    ----------
    input : ([batch], [channel], *inshape) tensor
        Input image.
    grid : ([batch], *outshape, dim) tensor
        Transformation field.
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : ([batch], [channel], *outshape) tensor
        Deformed image.

    """
    # Broadcast
    dim = grid.shape[-1]
    input_no_batch = input.dim() < dim + 2
    input_no_channel = input.dim() == dim
    grid_no_batch = grid.dim() == dim + 1
    if input_no_channel:
        input = input[None, None]
    elif input_no_batch:
        input = input[None]
    if grid_no_batch:
        grid = grid[None]
    batch = max(input.shape[0], grid.shape[0])
    channel = input.shape[1]
    input = expand(input, [batch, *input.shape[1:]])
    grid = expand(grid, [batch, *grid.shape[1:]])

    is_label = not utils.dtypes.dtype(input.dtype).is_floating_point
    if is_label:
        # label map -> specific processing
        out = input.new_zeros([batch, channel, *grid.shape[1:-1]])
        pmax = grid.new_zeros([batch, channel, *grid.shape[1:-1]])
        for label in input.unique():
            soft = (input == label).to(grid.dtype)
            soft = expand(soft, [batch, *input.shape[1:]])
            soft = GridPull.apply(soft, grid, interpolation, bound, extrapolate)
            out[soft > pmax] = label
            pmax = torch.max(pmax, soft)
    else:
        input = expand(input, [batch, *input.shape[1:]])
        out = GridPull.apply(input, grid, interpolation, bound, extrapolate)
    if input_no_channel:
        out = out[:, 0]
    if input_no_batch and grid_no_batch:
        out = out[0]
    return out


def grid_push(input, grid, shape=None, interpolation='linear', bound='zero',
              extrapolate=False):
    """Splat an image with respect to a deformation field (pull adjoint).

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    input : ([batch], [channel], *inshape) tensor
        Input image.
    grid : ([batch], *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : ([batch], [channel], *shape) tensor
        Spatted image.

    """
    # Broadcast
    dim = grid.shape[-1]
    input_no_batch = input.dim() == dim + 1
    input_no_channel = input.dim() == dim
    grid_no_batch = grid.dim() == dim + 1
    if input_no_channel:
        input = input[None, None]
    elif input_no_batch:
        input = input[None]
    if grid_no_batch:
        grid = grid[None]
    batch = max(input.shape[0], grid.shape[0])
    channel = input.shape[1]
    ndims = grid.shape[-1]
    input_shape = input.shape[2:]
    grid_shape = grid.shape[1:-1]
    spatial = [max(sinp, sgrd) for sinp, sgrd in zip(input_shape, grid_shape)]
    input = expand(input, [batch, channel, *spatial])
    grid = expand(grid, [batch, *spatial, ndims])

    if shape is None:
        shape = tuple(input.shape[2:])

    out = GridPush.apply(input, grid, shape, interpolation, bound, extrapolate)
    if input_no_channel:
        out = out[:, 0]
    if input_no_batch and grid_no_batch:
        out = out[0]
    return out


def grid_count(grid, shape=None, interpolation='linear', bound='zero',
               extrapolate=False):
    """Splatting weights with respect to a deformation field (pull adjoint).

    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    grid : ([batch], *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : ([batch], 1, *shape) tensor
        Spatting weights.

    """
    dim = grid.shape[-1]
    grid_no_batch = grid.dim() == dim + 1
    if grid_no_batch:
        grid = grid[None]
    if shape is None:
        shape = tuple(grid.shape[1:-1])

    out = GridCount.apply(grid, shape, interpolation, bound, extrapolate)
    if grid_no_batch:
        out = out[0]
    return out


def grid_grad(input, grid, interpolation='linear', bound='zero',
              extrapolate=False):
    """Sample spatial gradients of an image with respect to a deformation field.
    
    Notes
    -----
    {interpolation}

    {bound}

    Parameters
    ----------
    input : ([batch], [channel], *inshape) tensor
        Input image.
    grid : ([batch], *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    interpolation : int or sequence[int], default=1
        Interpolation order.
    bound : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : ([batch], [channel], *shape, dim) tensor
        Sampled gradients.

    """
    # Broadcast
    dim = grid.shape[-1]
    input_no_batch = input.dim() == dim + 1
    input_no_channel = input.dim() == dim
    grid_no_batch = grid.dim() == dim + 1
    if input_no_channel:
        input = input[None, None]
    elif input_no_batch:
        input = input[None]
    if grid_no_batch:
        grid = grid[None]
    batch = max(input.shape[0], grid.shape[0])
    input = expand(input, [batch, *input.shape[1:]])
    grid = expand(grid, [batch, *grid.shape[1:]])

    out = GridGrad.apply(input, grid, interpolation, bound, extrapolate)
    if input_no_channel:
        out = out[:, 0]
    if input_no_batch and grid_no_batch:
        out = out[0]
    return out


grid_pull.__doc__ = grid_pull.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
grid_push.__doc__ = grid_push.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
grid_count.__doc__ = grid_count.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)
grid_grad.__doc__ = grid_grad.__doc__.format(
    interpolation=_doc_interpolation, bound=_doc_bound)

# aliases
pull = grid_pull
push = grid_push
count = grid_count


def identity_grid(shape, dtype=None, device=None, jitter=False):
    """Returns an identity deformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial dimension of the field.
    dtype : torch.dtype, default=`get_default_dtype()`
        Data type.
    device torch.device, optional
        Device.
    jitter : bool or 'reproducible', default=False
        Jitter identity grid.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    mesh1d = [torch.arange(float(s), dtype=dtype, device=device)
              for s in shape]
    grid = torch.meshgrid(*mesh1d)
    grid = torch.stack(grid, dim=-1)
    if jitter:
        reproducible = jitter == 'reproducible'
        device_ids = [grid.device.index] if grid.device.type == 'cuda' else None
        with torch.random.fork_rng(device_ids, enabled=reproducible):
            if reproducible:
                torch.manual_seed(0)
            grid += torch.rand_like(grid)
            grid -= 0.5
    return grid


def affine_grid(mat, shape, jitter=False):
    """Create a dense transformation grid from an affine matrix.

    Parameters
    ----------
    mat : (..., D[+1], D[+1]) tensor
        Affine matrix (or matrices).
    shape : (D,) sequence[int]
        Shape of the grid, with length D.
    jitter : bool or 'reproducible', default=False
        Jitter identity grid.

    Returns
    -------
    grid : (..., *shape, D) tensor
        Dense transformation grid

    """
    mat = torch.as_tensor(mat)
    shape = list(shape)
    nb_dim = mat.shape[-1] - 1
    if nb_dim != len(shape):
        raise ValueError('Dimension of the affine matrix ({}) and shape ({}) '
                         'are not the same.'.format(nb_dim, len(shape)))
    if mat.shape[-2] not in (nb_dim, nb_dim+1):
        raise ValueError('First argument should be matrces of shape '
                         '(..., {0}, {1}) or (..., {1], {1}) but got {2}.'
                         .format(nb_dim, nb_dim+1, mat.shape))
    batch_shape = mat.shape[:-2]
    grid = identity_grid(shape, mat.dtype, mat.device, jitter=jitter)
    if batch_shape:
        grid = utils.unsqueeze(grid, dim=0, ndim=len(batch_shape))
        mat = utils.unsqueeze(mat, dim=-3, ndim=nb_dim)
    lin = mat[..., :nb_dim, :nb_dim]
    off = mat[..., :nb_dim, -1]
    grid = linalg.matvec(lin, grid) + off
    return grid


def resize(image, factor=None, shape=None, affine=None, anchor='c',
           *args, **kwargs):
    """Resize an image by a factor or to a specific shape.

    Notes
    -----
    .. A least one of `factor` and `shape` must be specified
    .. If `anchor in ('centers', 'edges')`, and both `factor` and `shape`
       are specified, `factor` is discarded.
    .. If `anchor in ('first', 'last')`, `factor` must be provided even
       if `shape` is specified.
    .. Because of rounding, it is in general not assured that
       `resize(resize(x, f), 1/f)` returns a tensor with the same shape as x.

        edges          centers          first           last
    e - + - + - e   + - + - + - +   + - + - + - +   + - + - + - +
    | . | . | . |   | c | . | c |   | f | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | . | . | . |   | . | . | . |   | . | . | . |
    + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +
    | . | . | . |   | c | . | c |   | . | . | . |   | . | . | l |
    e _ + _ + _ e   + _ + _ + _ +   + _ + _ + _ +   + _ + _ + _ +

    Parameters
    ----------
    image : (batch, channel, ...) tensor
        Image to resize
    factor : float or list[float], optional
        Resizing factor
        * > 1 : larger image <-> smaller voxels
        * < 1 : smaller image <-> larger voxels
    shape : (ndim,) sequence[int], optional
        Output shape
    affine : (batch, ndim[+1], ndim+1), optional
        Orientation matrix of the input image.
        If provided, the orientation matrix of the resized image is
        returned as well.
    anchor : {'centers', 'edges', 'first', 'last'} or list, default='centers'
        * In cases 'c' and 'e', the volume shape is multiplied by the
          zoom factor (and eventually truncated), and two anchor points
          are used to determine the voxel size.
        * In cases 'f' and 'l', a single anchor point is used so that
          the voxel size is exactly divided by the zoom factor.
          This case with an integer factor corresponds to subslicing
          the volume (e.g., `vol[::f, ::f, ::f]`).
        * A list of anchors (one per dimension) can also be provided.
    **kwargs : dict
        Parameters of `grid_pull`.

    Returns
    -------
    resized : (batch, channel, ...) tensor
        Resized image.
    affine : (batch, ndim[+1], ndim+1) tensor, optional
        Orientation matrix

    """
    # TODO: we could also use dft/dct/dst to resize, which correspond
    #   to some sort of sinc interpolation.

    # read parameters
    image = torch.as_tensor(image)
    nb_dim = image.dim() - 2
    inshape = image.shape[2:]
    info = {'dtype': image.dtype, 'device': image.device}
    factor = make_vector(factor or 0., nb_dim).tolist()
    outshape = make_list(shape or [None], nb_dim)
    anchor = [a[0].lower() for a in make_list(anchor, nb_dim)]
    return_trf = kwargs.pop('_return_trf', False)  # hidden option

    # compute output shape
    outshape = [o or int(i*f) for i, o, f in zip(inshape, outshape, factor)]
    if any(not s for s in outshape):
        raise ValueError('Either factor or shape must be set in '
                         'all dimensions')
    factor = [f or o/i for o, i, f in zip(outshape, inshape, factor)]

    # compute transformation grid
    # there is an affine relationship between the input and output grid:
    #    input_grid = scale * output_grid + shift
    lin = []
    scales = []
    shifts = []
    for anch, f, inshp, outshp in zip(anchor, factor, inshape, outshape):
        if anch == 'c':    # centers
            lin.append(torch.linspace(0, inshp - 1, outshp, **info))
            scales.append((inshp - 1) / (outshp - 1))
            shifts.append(0)
        elif anch == 'e':  # edges
            shift = (inshp * (1 / outshp - 1) + (inshp - 1)) / 2
            scale = inshp/outshp
            lin.append(torch.arange(0., outshp, **info) * scale + shift)
            scales.append(scale)
            shifts.append(shift)
        elif anch == 'f':  # first voxel
            lin.append(torch.arange(0., outshp, **info) / f)
            scales.append(1 / f)
            shifts.append(0)
        elif anch == 'l':  # last voxel
            shift = (inshp - 1) - (outshp - 1) / f
            lin.append(torch.arange(0., outshp, **info) / f + shift)
            scales.append(1 / f)
            shifts.append(shift)
        else:
            raise ValueError('Unknown anchor {}'.format(anch))
    grid = torch.stack(torch.meshgrid(*lin), dim=-1)[None, ...]

    # resize input image
    resized = grid_pull(image, grid, *args, **kwargs)

    # compute orientation matrix
    if affine is not None:
        affine, _ = affine_resize(affine, inshape, factor, anchor)
        if return_trf:
            return resized, affine, (scales, shifts)
        else:
            return resized, affine

    if return_trf:
        return resized, (scales, shifts)
    else:
        return resized


def resize_grid(grid, factor=None, shape=None, type='grid',
                affine=None, *args, **kwargs):
    """Resize a displacement grid by a factor.

    The displacement grid is resized *and* rescaled, so that
    displacements are expressed in the new voxel referential.

    Notes
    -----
    .. A least one of `factor` and `shape` must be specified.
    .. If `anchor in ('centers', 'edges')`, and both `factor` and `shape`
       are specified, `factor` is discarded.
    .. If `anchor in ('first', 'last')`, `factor` must be provided even
       if `shape` is specified.
    .. Because of rounding, it is in general not assured that
       `resize(resize(x, f), 1/f)` returns a tensor with the same shape as x.

    Parameters
    ----------
    grid : (batch, ..., ndim) tensor
        Grid to resize
    factor : float or list[float], optional
        Resizing factor
        * > 1 : larger image <-> smaller voxels
        * < 1 : smaller image <-> larger voxels
    shape : (ndim,) sequence[int], optional
        Output shape
    type : {'grid', 'displacement'}, default='grid'
        Grid type:
        * 'grid' correspond to dense grids of coordinates.
        * 'displacement' correspond to dense grid of relative displacements.
        Both types are not rescaled in the same way.
    affine : (batch, ndim[+1], ndim+1), optional
        Orientation matrix of the input grid.
        If provided, the orientation matrix of the resized image is
        returned as well.
    anchor : {'centers', 'edges', 'first', 'last'}, default='centers'
        * In cases 'c' and 'e', the volume shape is multiplied by the
          zoom factor (and eventually truncated), and two anchor points
          are used to determine the voxel size.
        * In cases 'f' and 'l', a single anchor point is used so that
          the voxel size is exactly divided by the zoom factor.
          This case with an integer factor corresponds to subslicing
          the volume (e.g., `vol[::f, ::f, ::f]`).
        * A list of anchors (one per dimension) can also be provided.
    **kwargs
        Parameters of `grid_pull`.

    Returns
    -------
    resized : (batch, ..., ndim) tensor
        Resized grid.
    affine : (batch, ndim[+1], ndim+1) tensor, optional
        Orientation matrix

    """
    # resize grid
    kwargs['_return_trf'] = True
    grid = utils.last2channel(grid)
    outputs = resize(grid, factor, shape, affine, *args, **kwargs)
    if affine is not None:
        grid, affine, (scales, shifts) = outputs
    else:
        grid, (scales, shifts) = outputs
    grid = utils.channel2last(grid)

    # rescale each component
    # scales and shifts map resized coordinates to original coordinates:
    #   original = scale * resized + shift
    # here we want to transform original coordinates into resized ones:
    #   resized = (original - shift) / scale
    grids = []
    for d, (scl, shft) in enumerate(zip(scales, shifts)):
        grid1 = utils.slice_tensor(grid, d, dim=-1)
        if type[0].lower() == 'g':
            grid1 = grid1 - shft
        grid1 = grid1 / scl
        grids.append(grid1)
    grid = torch.stack(grids, -1)

    # return
    if affine is not None:
        return grid, affine
    else:
        return grid


def reslice(image, affine, affine_to, shape_to=None, **kwargs):
    """Reslice a spatial image to a different space (shape + affine).

    Parameters
    ----------
    image : (*batch, *channels, *spatial)
        Input image
    affine : (*batch, dim[+1], dim+1)
        Input affine
    affine_to : (*batch, dim[+1], dim+1)
        Target affine
    shape_to : (dim,) sequence[int], optional
        Target shape. Default: same as input shape

    Other Parameters
    ----------------
    Parameters of `grid_pull`

    Returns
    -------
    resliced : (*batch, *channels, *shape_to)
        Resliced image.

    """
    # prepare tensors
    image = torch.as_tensor(image)
    backend = dict(dtype=image.dtype, device=image.device)
    affine = torch.as_tensor(affine, **backend)
    affine_to = torch.as_tensor(affine_to, **backend)

    # compute shape components
    dim = affine.shape[-1] - 1
    batch = affine.shape[:-2]
    channels = image.shape[len(batch):-dim]
    shape = image.shape[-dim:]
    if shape_to is None:
        shape_to = shape

    # perform reslicing
    #   > image must be reshaped to (B, C, *spatial) for grid_pull
    transformation = affine_lmdiv(affine, affine_to)        # (*batch, d+1, d+1)
    grid = affine_grid(transformation, shape_to)            # (*batch, *shape_to, d)
    if not prod(batch):
        grid = grid[None]                                   # (*batch|1, *shape_to, d)
    squeeze_shape = [prod(batch) or 1, prod(channels) or 1, *shape]
    image = image.reshape(squeeze_shape)                    # (b, c, *spatial)
    image = grid_pull(image, grid, **kwargs)                # (b, c, *shape_to)
    image = image.reshape([*batch, *channels, *shape_to])   # (*batch, *channels, *shape_to)
    return image


def grid_inv(grid, type='grid', lam=0.1, bound='dft',
             extrapolate=True):
    """Invert a dense deformation (or displacement) grid
    
    Notes
    -----
    The deformation/displacement grid must be expressed in 
    voxels, and map from/to the same lattice.
    
    Let `f = id + d` be the transformation. The inverse 
    is obtained as `id - (f.T @ 1 + L) \ (f.T @ d)`
    where `L` is a regulariser, `f.T @ _` is the adjoint 
    operation ("push") of `f @ _` ("pull"). and `1` is an 
    image of ones. 
    
    The idea behind this is that `f.T @ _` is approximately
    the inverse transformation weighted by the determinant
    of the Jacobian of the tranformation so, in the (theoretical)
    continuous case, `inv(f) @ _ = f.T @ _ / f.T @ 1`.
    However, in the (real) discrete case, this leads to 
    lots of holes in the inverse. The solution we use 
    therefore fills these holes using a regularised 
    least-squares scheme, where the regulariser penalizes
    the spatial gradients of the inverse field.
    
    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation (or displacement) grid
    type : {'grid', 'disp'}, default='grid'
        Type of deformation.
    lam : float, default=0.1
        Regularisation
    bound : str, default='dft'
        Boundary conditions
    extrapolate : bool, default=True
        Extrapolate the transformation field when
        it is sampled out-of-bounds.
        
    Returns
    -------
    grid_inv : (..., *spatial, dim)
        Inverse transformation (or displacement) grid
    
    """
    # get shape components
    grid = torch.as_tensor(grid)
    dim = grid.shape[-1]
    shape = grid.shape[-(dim+1):-1]
    batch = grid.shape[:-(dim+1)]
    grid = grid.reshape([-1, *shape, dim])
    backend = dict(dtype=grid.dtype, device=grid.device)
    
    # get displacement
    identity = identity_grid(shape, **backend)
    if type == 'grid':
        disp = grid - identity
    else:
        disp = grid
        grid = disp + identity
    
    # push displacement
    push_opt = dict(bound=bound, extrapolate=extrapolate)
    disp = utils.movedim(disp, -1, 1)
    disp = grid_push(disp, grid, **push_opt)
    count = grid_count(grid, **push_opt)
    disp = utils.movedim(disp, 1, -1)
    count = utils.movedim(count, 1, -1)
    
    # Fill missing values using regularised least squares
    disp = solve_grid_sym(count, disp, bound=bound,
                          membrane=lam)
    disp = disp.reshape([*batch, *shape, dim])
    
    if type == 'grid':
        return identity - disp
    else:
        return -disp

def grid_jacobian(grid, bound='dft', voxel_size=1, type='grid'):
    """Compute the Jacobian of a transformation field

    Notes
    -----
    .. If a displacement (`type='disp'`) is provided, we compute the Jacobian
       of the transformation field (identity + displacement) by
       adding ones to the diagonal.
    .. This function uses central finite differences to estimate the
       Jacobian.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation ('grid') or displacement ('disp')
    bound : str, default='dft'
    voxel_size : [sequence of] float, default=1
    type : {'grid', 'disp'}, default='grid'

    Returns
    -------
    jac : (..., *spatial, dim, dim) tensor
        Jacobian. In each matrix: jac[i, j] = d psi[i] / d xj

    """
    grid = torch.as_tensor(grid)
    dim = grid.shape[-1]
    if type == 'grid':
        grid -= identity_grid(grid.shape[-dim-1:-1], **utils.backend(grid))
    dims = list(range(-dim-1, -1))
    jac = diff(grid, dim=dims, bound=bound, voxel_size=voxel_size, side='c')
    torch.diagonal(jac, 0, -1, -2).add_(1)
    return jac


def grid_jacdet(grid, bound='dft', voxel_size=1, type='grid'):
    """Compute the Jacobian determinant of a transformation field

    Notes
    -----
    .. If a displacement (`type='disp'`) is provided, we compute the Jacobian
       of the transformation field (identity + displacement) by
       adding ones to the diagonal.
    .. This function uses central finite differences to estimate the
       Jacobian.

    Parameters
    ----------
    grid : (..., *spatial, dim) tensor
        Transformation ('grid') or displacement ('disp')
    bound : str, default='dft'
    voxel_size : [sequence of] float, default=1
    type : {'grid', 'disp'}, default='grid'

    Returns
    -------
    det : (..., *spatial) tensor
        Jacobian determinant.

    """
    jac = grid_jacobian(grid, bound=bound, voxel_size=voxel_size, type=type)
    return jac.det()


# def transform_points(points, grid, type='grid',
#                      affine=None, points_unit='mm', grid_unit='voxels',
#                      bound='zero', interpolation=1, extrapolate=0):
#     """
#
#     Parameters
#     ----------
#     points : ([batch], collection, dim) tensor
#         Collection of points (in
#     grid : ([batch], *spatial, dim) tensor
#         Transformation of displacement grid
#     type : {'grid', 'displacement'}
#     affine
#     bound
#     interpolation
#     extrapolate
#
#     Returns
#     -------
#
#     """
#
#     vertices = torch.as_tensor(vertices)
#     disp = torch.as_tensor(disp)
#     aff = torch.as_tensor(aff)
#     # convert vertices to voxels to sample
#     grid = ni.spatial.affine_matvec(ni.spatial.affine_inv(aff),
#                                          vertices)
#     grid = grid.reshape([1, 1, -1, 3])  # make 3d
#     # sample displacement
#     wdisp = ni.spatial.grid_pull(disp.movedim(-1, 0), grid,
#                                       bound='zero', extrapolate=False)
#     wdisp = wdisp.movedim(0, -1)
#     grid = grid + wdisp
#     grid = grid.reshape([-1, 3])
#     # convert voxels to mm
#     wvertices = ni.spatial.affine_matvec(aff, grid)