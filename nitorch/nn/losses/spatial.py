"""
Losses that assume an underlying spatial organization
(gradients, curvature, etc.)
"""

import torch
import torch.nn as tnn
from nitorch.core.py import make_list, prod
from nitorch.core.utils import slice_tensor
from nitorch.spatial import diff1d
from .base import Loss


class LocalFeatures(tnn.Module):
    """Base class for feature extractors.

    Is it really useful?
    """
    def __init__(self, bound='dct2', voxel_size=1, *args, **kwargs):
        """

        Parameters
        ----------
        bound : BoundType, default='dct2'
            Boundary conditions, used to compute derivatives at the edges.
        voxel_size : float or list[float], default=1
            Voxel size
        """
        super().__init__(*args, **kwargs)
        self.bound = bound
        self.voxel_size = voxel_size


class Diff(LocalFeatures):
    """Finite differences."""

    def __init__(self, order=1, side='c', dim=None, *args, **kwargs):
        """

        Parameters
        ----------
        order : int, default=1
            Finite differences order
        side : {'c', 'f', 'b'} or list[{'c', 'f', 'b'}], default='c'
            Type of finite-differencesto extract about each voxel:
                * 'c' : central  -> `g[i] = (x[i+1] - x[i-1])/2`
                * 'f' : forward  -> `g[i] = (x[i+1] - x[i])`
                * 'b' : backward -> `g[i] = (x[i] - x[i-1])`
        dim : int or list[int], optional
            Dimensions along which to compute the finite differences.
            By default, all except the first two (batch and channel).
        bound : BoundType or list[BoundType], default='dct2'
            Boundary conditions, used to compute derivatives at the edges.
        voxel_size : float or list[float], default=1
            Voxel size
        reduction : {'mean', 'sum'} or callable, default='mean'
            Type of reduction to apply.
        """
        super().__init__(*args, **kwargs)
        self.order = order
        self.side = side
        self.dim = dim

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : tensor
            Input tensor with shape (batch, channel, *spatial)
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        g : tensor
            Finite differences with shape
            (batch, channel, *spatial, len(dim), len(side))

            If `dim` or `side` are scalars, not lists, their respective
            dimension is dropped in the output tensor.
            E.g., if `side='c'`, the output shape is
            (batch, channel, *spatial, len(dim))

        """
        order = overload.get('order', self.order)
        side = make_list(overload.get('side', self.side))
        drop_side_dim = not isinstance(side, (tuple, list))
        side = make_list(side)
        dim = overload.get('dim', self.dim)
        dim = list(range(2, x.dim())) if dim is None else dim
        drop_dim_dim = not isinstance(dim, (tuple, list))
        dim = make_list(dim)
        nb_dim = len(dim)
        voxel_size = overload.get('voxel_size',  self.voxel_size)
        voxel_size = make_list(voxel_size, nb_dim)
        bound = make_list(overload.get('bound', self.bound), nb_dim)

        diffs = []
        for d, vx, bnd in zip(dim, voxel_size, bound):
            sides = []
            for s in side:
                grad = diff1d(x, order=order, dim=d, voxel_size=vx,
                              side=s, bound=bnd)
                sides.append(grad)
            sides = torch.stack(sides, dim=-1)
            diffs.append(sides)
        diffs = torch.stack(diffs, dim=-2)

        if drop_dim_dim:
            diffs = slice_tensor(diffs, 0, dim=-2)
        if drop_side_dim:
            diffs = slice_tensor(diffs, 0, dim=-1)
        return diffs


class MembraneLoss(Loss):
    """Compute the membrane energy (squared gradients) of a tensor.

    The membrane energy of a field is the integral of its squared
    gradient magnitude (l2 norm). This class extends this concept to
    other norms of the gradient (l1, l{1,2}).

    In the l2 case, if we name "f" the unit of the field and "m" the
    spatial unit of a voxel, the output loss has unit `(f/m)**2`.
    If `factor` is used to weight each voxel by its volume (as should
    be done in a proper integration) the unit becomes
    `(f/m)**2 * m**d =  f**2 * m**(d-2)`.

    In the l1 case, it is `f/m` in the absence of weighting and
    `f * m**(d-1)` with volume weighting.

    """

    def __init__(self, voxel_size=1, factor=1, bound='dct2', l1=None,
                 *args, **kwargs):
        """

        Parameters
        ----------
        voxel_size : float or list[float], default=1
            Voxel size. Useful for anisotropic tensors (where the
            sampling rate is higher in some directions than others).
        factor : float or list[float], default=1
            Scale the loss by a per-dimension factor. Useful when
            working with resized tensor to compensate for different
            number of voxels.
        bound : BoundType, default='dct2'
            Boundary conditions, used to compute derivatives at the edges.
        l1 : bool or int or list[int], default=None
            Dimensions along which to apply a square root reduction
            ('l1 norm'), after taking the square. Dimensions are
            those of the gradient map with shape
            (batch, channel, *spatial, direction, side)
                * False: nowhere == (squared) l2 norm
                * True: everywhere == l1 norm
                * Otherwise: l_{1,2} norm (group sparsity)

        """
        super().__init__(*args, **kwargs)
        self.voxel_size = voxel_size
        self.factor = factor
        self.bound = bound
        self.l1 = l1

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : tensor
            Input tensor
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        nb_dim = x.dim() - 2
        voxel_size = make_list(overload.get('voxel_size', self.voxel_size), nb_dim)
        factor = make_list(overload.get('factor', self.factor), nb_dim)
        bound = make_list(overload.get('bound', self.bound), nb_dim)
        l1 = overload.get('l1', self.l1)

        # Compute spatial gradients
        #
        # TODO: when penalty == 'l2', for some boundary conditions, there's no
        #   need to compute both forward and backward gradients as they are
        #   the same (but shifted). For now, to avoid having to detect which
        #   cases can be accelerated, I always compute both (more general).
        loss = Diff(side=['f', 'b'], bound=bound, voxel_size=voxel_size)(x)
        loss = loss.square()

        # Apply l1
        if l1 not in (None, False):
            if l1 is True:
                loss = loss.sqrt()
            else:
                l1 = make_list(l1)
                loss = loss.sum(dim=l1).sqrt()  # TODO: use self.reduction instead of sum?

        # Reduce
        loss = super().forward(loss)

        # Scale
        factor = prod(factor)
        if factor != 1:
            loss = loss * factor

        return loss


class BendingLoss(Loss):
    """Compute the bending energy (squared gradients) of a tensor.

    The bending energy of a field is the integral of its squared
    second-order derivatives magnitude (l2 norm).
    This class extends this concept to other norms of the gradient
    (l1, l{1,2}).

    In the l2 case, if we name "f" the unit of the field and "m" the
    spatial unit of a voxel, the output loss has unit `(f/m**2)**2`.
    If `factor` is used to weight each voxel by its volume (as should
    be done in a proper integration) the unit becomes
    `(f/m**2)**2 * m**d =  f**2 * m**(d-4)`.

    In the l1 case, it is `f/m**2` in the absence of weighting and
    `f * m**(d-2)` with volume weighting.

    """

    def __init__(self, voxel_size=1, factor=1, bound='dct2', l1=None,
                 *args, **kwargs):
        """

        Parameters
        ----------
        voxel_size : float or list[float], default=1
            Voxel size. Useful for anisotropic tensors (where the
            sampling rate is higher in some directions than others).
        factor : float or list[float], default=1
            Scale the loss by a per-dimension factor. Useful when
            working with resized tensor to compensate for different
            number of voxels.
        bound : BoundType, default='dct2'
            Boundary conditions, used to compute derivatives at the edges.
        l1 : bool or int or list[int], default=None
            Dimensions along which to apply a square root reduction
            ('l1 norm'), after taking the square. Dimensions are
            those of the gradient map with shape
            (batch, channel, *spatial, direction)
                * False: nowhere == (squared) l2 norm
                * True: everywhere == l1 norm
                * Otherwise: l_{1,2} norm (group sparsity)

        """
        super().__init__(*args, **kwargs)
        self.voxel_size = voxel_size
        self.factor = factor
        self.bound = bound
        self.l1 = l1

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : tensor
            Input tensor
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        nb_dim = x.dim() - 2
        voxel_size = make_list(overload.get('voxel_size', self.voxel_size), nb_dim)
        factor = make_list(overload.get('factor', self.factor), nb_dim)
        bound = make_list(overload.get('bound', self.bound), nb_dim)
        l1 = overload.get('l1', self.l1)

        # Compute spatial gradients
        loss = Diff(order=2, side='c', bound=bound, voxel_size=voxel_size)(x)
        loss = loss.square()

        # Apply l1
        if l1 not in (None, False):
            if l1 is True:
                loss = loss.sqrt()
            else:
                l1 = make_list(l1)
                loss = loss.sum(dim=l1).sqrt()

        # Reduce
        loss = super().forward(loss)

        # Scale
        factor = prod(factor)
        if factor != 1:
            loss = loss * factor

        return loss


class LameShearLoss(Loss):
    """Strain-part of the (Linear)-Elastic energy (penalty on shears).

    = second Lame constant = shear modulus

    The shear energy of a deformation field is the integral of the square
    magnitude (l2 norm) of the symetric part diagonal terms of its Jacobian.
    This class extends this concept to  other norms of the gradient
    (l1, l{1,2}).

    In the l2 case, E = sum_{i != j} (dv[i]/dx[j]) ** 2.

    """

    def __init__(self, voxel_size=1, factor=1, bound='dct2', l1=None,
                 exclude_zooms=False, *args, **kwargs):
        """

        Parameters
        ----------
        voxel_size : float or list[float], default=1
            Voxel size. Useful for anisotropic tensors (where the
            sampling rate is higher in some directions than others).
        factor : float or list[float], default=1
            Scale the loss by a per-dimension factor. Useful when
            working with resized tensor to compensate for different
            number of voxels.
        bound : BoundType, default='dct2'
            Boundary conditions, used to compute derivatives at the edges.
        l1 : bool or int or list[int], default=None
            Dimensions along which to apply a square root reduction
            ('l1 norm'), after taking the square. Dimensions are
            those of the gradient map with shape
            (batch, channel, *spatial, side)
                * False: nowhere == (squared) l2 norm
                * True: everywhere == l1 norm
                * Otherwise: l_{1,2} norm (group sparsity)

            Here, `channel` map to elements of the Jacobian matrix, while
            `side` map to the combination of sides (forward/backward)
            used when extracting finite differences. Therefore, the
            number of channels is dim*(dim+1)//2 and the number of sides
            is 4.
        exclude_zooms : bool, default=False
            Do not include diagonal elements of the Jacobian in the
            penalty (i.e., penalize only shears)

        """
        super().__init__(*args, **kwargs)
        self.voxel_size = voxel_size
        self.factor = factor
        self.bound = bound
        self.l1 = l1
        self.exclude_zooms = exclude_zooms

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : (batch, ndim, *spatial) tensor
            Input displacement tensor (in channel first order)
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        nb_dim = x.dim() - 2
        voxel_size = make_list(overload.get('voxel_size', self.voxel_size), nb_dim)
        factor = make_list(overload.get('factor', self.factor), nb_dim)
        bound = make_list(overload.get('bound', self.bound), nb_dim)
        l1 = overload.get('l1', self.l1)
        exclude_zooms = overload.get('exclude_zooms', self.exclude_zooms)

        # Compute spatial gradients
        loss_diag = []      # diagonal elements of the Jacobian
        loss_offdiag = []   # off-diagonal elements of hte (symmetric) Jacobian
        for i in range(nb_dim):
            # symmetric part
            x_i = x[:, i:i+1, ...]
            subloss_diag = []
            subloss_offdiag = []
            for j in range(nb_dim):
                for side_i in ('f', 'b'):
                    diff = Diff(dim=[j+2], side=side_i, bound=bound,
                                voxel_size=voxel_size)
                    diff_ij = diff(x_i)
                    if i == j:
                        # diagonal elements
                        if not exclude_zooms:
                            subloss_diag.append(diff_ij)
                    else:
                        # off diagonal elements
                        x_j = x[:, j:j+1, ...]
                        for side_j in ('f', 'b'):
                            diff = Diff(dim=[i+2], side=side_j, bound=bound,
                                        voxel_size=voxel_size)
                            diff_ji = diff(x_j)
                            subloss_offdiag.append((diff_ij + diff_ji)/2)
            if not exclude_zooms:
                loss_diag.append(torch.stack(subloss_diag, dim=-1))
            loss_offdiag.append(torch.stack(subloss_offdiag, dim=-1))
        if not exclude_zooms:
            loss_diag = torch.cat(loss_diag, dim=1)
        loss_offdiag = torch.cat(loss_offdiag, dim=1)

        if l1 not in (None, False):
            # Apply l1 reduction
            if l1 is True:
                if not exclude_zooms:
                    loss_diag = loss_diag.abs()
                loss_offdiag = loss_offdiag.abs()
            else:
                l1 = make_list(l1)
                if not exclude_zooms:
                    loss_diag = loss_diag.square().sum(dim=l1, keepdim=True).sqrt()
                loss_offdiag = loss_offdiag.square().sum(dim=l1, keepdim=True).sqrt()
        else:
            # Apply l2 reduction
            if not exclude_zooms:
                loss_diag = loss_diag.square()
            loss_offdiag = loss_offdiag.square()

        # Mean reduction across sides
        if not exclude_zooms:
            loss_diag = loss_diag.mean(dim=-1)
        loss_offdiag = loss_offdiag.mean(dim=-1)

        # Weighted reduction across elements
        if not exclude_zooms:
            if loss_diag.shape[1] == 1:
                # element dimension already reduced -> we need a small hack
                loss = (loss_diag.square() + 2*loss_offdiag.square()) / (nb_dim**2)
                loss = loss.sum(dim=1, keepdim=True).sqrt()
            else:
                # simple weighted average
                loss = (loss_diag.sum(dim=1, keepdim=True) +
                        loss_offdiag.sum(dim=1, keepdim=True)*2) / (nb_dim**2)
        else:
            loss = loss_offdiag.sum(dim=1, keepdim=True)*2 / (nb_dim**2)

        # Reduce
        loss = super().forward(loss)

        # Scale
        factor = prod(factor)
        if factor != 1:
            loss = loss * factor

        return loss


class LameZoomLoss(Loss):
    """Compression-part of the (Linear)-Elastic energy (penalty on volume change).

    = first Lame constant

    The compression energy of a deformation field is the integral of the square
    magnitude (l2 norm) of the trace its Jacobian.
    This class extends this concept to  other norms of the gradient
    (l1, l{1,2}).

    In the l2 case, E = sum_{ij} (dv[i]/dx[j] + dv[j]/dx[i]) ** 2.

    """

    def __init__(self, voxel_size=1, factor=1, bound='dct2', l1=None,
                 *args, **kwargs):
        """

        Parameters
        ----------
        voxel_size : float or list[float], default=1
            Voxel size. Useful for anisotropic tensors (where the
            sampling rate is higher in some directions than others).
        factor : float or list[float], default=1
            Scale the loss by a per-dimension factor. Useful when
            working with resized tensor to compensate for different
            number of voxels.
        bound : BoundType, default='dct2'
            Boundary conditions, used to compute derivatives at the edges.
        l1 : bool or int or list[int], default=None
            Dimensions along which to apply a square root reduction
            ('l1 norm'), after taking the square. Dimensions are
            those of the gradient map with shape
            (batch, channel, *spatial, direction, side)
                * False: nowhere == (squared) l2 norm
                * True: everywhere == l1 norm
                * Otherwise: l_{1,2} norm (group sparsity)

        """
        super().__init__(*args, **kwargs)
        self.voxel_size = voxel_size
        self.factor = factor
        self.bound = bound
        self.l1 = l1

    def forward(self, x, **overload):
        """

        Parameters
        ----------
        x : tensor
            Input tensor
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        loss : scalar or tensor
            The output shape depends on the type of reduction used.
            If 'mean' or 'sum', this function returns a scalar.

        """

        nb_dim = x.dim() - 2
        voxel_size = make_list(overload.get('voxel_size', self.voxel_size), nb_dim)
        factor = make_list(overload.get('factor', self.factor), nb_dim)
        bound = make_list(overload.get('bound', self.bound), nb_dim)
        l1 = overload.get('l1', self.l1)

        # Compute spatial gradients
        loss = []
        for i in range(nb_dim):
            x_i = x[:, i:i+1, ...]
            diff = Diff(dim=[i], side=['f', 'b'], bound=bound,
                        voxel_size=voxel_size)
            loss.append(diff(x_i))
        loss = torch.cat(loss, dim=1)
        loss = loss.square()

        # Apply l1
        if l1 not in (None, False):
            if l1 is True:
                loss = loss.sqrt()
            else:
                l1 = make_list(l1)
                loss = loss.sum(dim=l1, keepdim=True).sqrt()

        # Mean reduction across sides
        loss = loss.mean(dim=-1)

        # Reduce
        loss = super().forward(loss)

        # Scale
        factor = prod(factor)
        if factor != 1:
            loss = loss * factor

        return loss
