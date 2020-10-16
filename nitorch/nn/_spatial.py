# -*- coding: utf-8 -*-
"""Spatial transformation layers."""

import torch
from torch import nn as tnn
from .. import spatial
from ._cnn import UNet
from ._base import Module


_interpolation_doc = \
    """`interpolation` can be an int, a string or an InterpolationType.
    Possible values are:
        - 0 or 'nearest'
        - 1 or 'linear'
        - 2 or 'quadratic'
        - 3 or 'cubic'
        - 4 or 'fourth'
        - 5 or 'fifth'
        - 6 or 'sixth'
        - 7 or 'seventh'
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific interpolation orders."""

_bound_doc = \
    """`bound` can be a string or a BoundType.
    Possible values are:
        - 'replicate'  or 'nearest'
        - 'dct1'       or 'mirror'
        - 'dct2'       or 'reflect'
        - 'dst1'       or 'antimirror'
        - 'dst2'       or 'antireflect'
        - 'dft'        or 'wrap'
        - 'zero'
    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific boundary conditions.
    Note that:
    - `dft` corresponds to circular padding
    - `dct2` corresponds to Neumann boundary conditions (symmetric)
    - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)
    See https://en.wikipedia.org/wiki/Discrete_cosine_transform
        https://en.wikipedia.org/wiki/Discrete_sine_transform"""


class GridPull(Module):
    __doc__ = """
    Pull/Sample an image according to a deformation.

    This module has no learnable parameters.

    {interpolation}

    {bound}
    """.format(interpolation=_interpolation_doc, bound=_bound_doc)

    def __init__(self, interpolation='linear', bound='dct2', extrapolate=True):
        """

        Parameters
        ----------
        interpolation : InterpolationType or list[InterpolationType], default=1
            Interpolation order.
        bound : BoundType, or list[BoundType], default='dct2'
            Boundary condition.
        extrapolate : bool, default=True
            Extrapolate data outside of the field of view.
        """
        super().__init__()
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, x, grid, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial_in) tensor
            Input image to deform
        grid : (batch, *spatial_out, len(spatial_in)) tensor
            Transformation grid
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        pulled : (batch, channel, *spatial_out) tensor
            Deformed image.

        """
        interpolation = overload.get('interpolation', self.interpolation)
        bound = overload.get('bound', self.bound)
        extrapolate = overload.get('extrapolate', self.extrapolate)
        return spatial.grid_pull(x, grid, interpolation, bound, extrapolate)


class GridPush(Module):
    __doc__ = """
    Push/Splat an image according to a deformation.

    This module has no learnable parameters.

    {interpolation}

    {bound}
    """.format(interpolation=_interpolation_doc, bound=_bound_doc)

    def __init__(self, shape=None, interpolation='linear', bound='dct2',
                 extrapolate=True):
        """

        Parameters
        ----------
        shape : list[int], optional
            Output spatial shape. Default is the same as the input shape.
        interpolation : InterpolationType or list[InterpolationType], default=1
            Interpolation order.
        bound : BoundType, or list[BoundType], default='dct2'
            Boundary condition.
        extrapolate : bool, default=True
            Extrapolate data outside of the field of view.
        """
        super().__init__()
        self.shape = shape
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, x, grid, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial_in) tensor
            Input image to deform
        grid : (batch, *spatial_out, len(spatial_in)) tensor
            Transformation grid
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        pushed : (batch, channel, *spatial_out) tensor
            Deformed image.

        """
        shape = overload.get('shape', self.shape)
        interpolation = overload.get('interpolation', self.interpolation)
        bound = overload.get('bound', self.bound)
        extrapolate = overload.get('extrapolate', self.extrapolate)
        return spatial.grid_push(x, grid, shape,
                                 interpolation=interpolation,
                                 bound=bound,
                                 extrapolate=extrapolate)


class GridPushCount(Module):
    __doc__ = """
    Push/Splat an image **and** ones according to a deformation.

    Both an input image and an image of ones of the same shape are pushed.
    The results are concatenated along the channel dimension.

    This module has no learnable parameters.

    {interpolation}

    {bound}
    """.format(interpolation=_interpolation_doc, bound=_bound_doc)

    def __init__(self, shape=None, interpolation='linear', bound='dct2',
                 extrapolate=True):
        """

        Parameters
        ----------
        shape : list[int], optional
            Output spatial shape. Default is the same as the input shape.
        interpolation : InterpolationType or list[InterpolationType], default=1
            Interpolation order.
        bound : BoundType, or list[BoundType], default='dct2'
            Boundary condition.
        extrapolate : bool, default=True
            Extrapolate data outside of the field of view.
        """
        super().__init__()
        self.shape = shape
        self.interpolation = interpolation
        self.bound = bound
        self.extrapolate = extrapolate

    def forward(self, x, grid, **overload):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial_in) tensor
            Input image to deform
        grid : (batch, *spatial_in, dir) tensor
            Transformation grid
        overload : dict
            All parameters defined at build time can be overridden
            at call time.

        Returns
        -------
        pushed : (batch, channel, *shape) tensor
            Pushed image.
        count : (batch, 1, *shape) tensor
            Pushed image.

        """
        shape = overload.get('shape', self.shape)
        interpolation = overload.get('interpolation', self.interpolation)
        bound = overload.get('bound', self.bound)
        extrapolate = overload.get('extrapolate', self.extrapolate)
        push = spatial.grid_push(x, grid, shape,
                                 interpolation=interpolation,
                                 bound=bound,
                                 extrapolate=extrapolate)
        count = spatial.grid_count(grid, shape,
                                   interpolation=interpolation,
                                   bound=bound,
                                   extrapolate=extrapolate)
        return push, count


class GridExp(Module):
    """Exponentiate an inifinitesimal deformation field (velocity)."""

    def __init__(self, fwd=True, inv=False, steps=None,
                 interpolation='linear', bound='dft', displacement=False,
                 energy=None, vs=None, greens=None, inplace=True):
        """

        Parameters
        ----------
        fwd : bool, default=True
            Return the forward deformation.
        inv : bool, default=False
            Return the inverse deformation.
        steps : int, optional
            Number of integration steps.
            Use `1` to use a small displacements model instead of a
            diffeomorphic one. Default is an educated guess based on the
            magnitude of the velocity field.
        interpolation : {0..7}, default=1
            Interpolation order. Can also be names ('nearest', 'linear', etc.).
        bound : {'dft', 'dct1', 'dct2', 'dst1', 'dst2'}, default='dft'
            Boundary conditions.
        displacement : bool, default=False
            Return a displacement field rather than a transformation field.
        energy : default=None
            If None: squaring and scaling integration.
        vs : list[float], default=1
            Voxel size.
        greens : tensor_like, optional
            Pre-computed Greens function (= inverse kernel in freq. domain)
        inplace : bool, default=True
            Perform the integration inplace if possible.
        """
        super().__init__()

        self.fwd = fwd
        self.inv = inv
        self.steps = steps
        self.interpolation = interpolation
        self.bound = bound
        self.displacement = displacement
        self.energy = energy
        self.vs = vs
        self.greens = greens
        self.inplace = inplace

    def forward(self, velocity, **kwargs):
        """

        Parameters
        ----------
        velocity (tensor) : velocity field with shape (batch, *spatial, dim).
        **kwargs : all parameters of the module can be overridden at call time.

        Returns
        -------
        forward (tensor, if `forward is True`) : forward displacement
            (if `displacement is True`) or transformation (if `displacement
            is False`) field, with shape (batch, *spatial, dim)
        inverse (tensor, if `inverse is True`) : forward displacement
            (if `displacement is True`) or transformation (if `displacement
            is False`) field, with shape (batch, *spatial, dim)

        """
        fwd = kwargs.get('fwd', self.forward)
        inv = kwargs.get('inverse', self.inv)
        steps = kwargs.get('steps', self.steps)
        interpolation = kwargs.get('interpolation', self.interpolation)
        bound = kwargs.get('bound', self.bound)
        displacement = kwargs.get('displacement', self.displacement)
        energy = kwargs.get('energy', self.energy)
        vs = kwargs.get('vs', self.vs)
        greens = kwargs.get('greens', self.greens)
        inplace = False  # kwargs.get('inplace', self.inplace)

        output = []
        if fwd:
            y = spatial.exp(velocity, False, steps, interpolation, bound,
                            displacement, energy, vs, greens, inplace)
            output.append(y)
        if inv:
            iy = spatial.exp(velocity, True, steps, interpolation, bound,
                             displacement, energy, vs, greens, inplace)
            output.append(iy)

        return output if len(output) > 1 else \
               output[0] if len(output) == 1 else \
               None


class VoxelMorph(Module):
    """VoxelMorph warps a source/moving image to a fixed/target image.

    A VoxelMorph network is obtained by concatenating a UNet and a
    (diffeomorphic) spatial transformer. The loss is made of two terms:
    an image similarity loss and a velocity regularisation loss.

    The UNet used here is slightly different from the original one (we
    use a fully convolutional network -- based on strided convolutions --
    instead of maxpooling and upsampling).

    References
    ----------
    .. [1] "An Unsupervised Learning Model for Deformable Medical Image Registration"
        Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
        CVPR 2018. eprint arXiv:1802.02604
    .. [2] "VoxelMorph: A Learning Framework for Deformable Medical Image Registration"
        Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
        IEEE TMI: Transactions on Medical Imaging. 2019. eprint arXiv:1809.05231
    .. [3] "Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MICCAI 2018. eprint arXiv:1805.04605
    .. [4] "Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces"
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
    """

    def __init__(self, dim, encoder=None, decoder=None, kernel_size=3,
                 interpolation='linear', grid_bound='dft', image_bound='dct2'):
        """

        Parameters
        ----------
        dim : int
            Dimensionalityy of the input (1|2|3)
        encoder : list[int], optional
            Number of channels after each encoding layer of the UNet.
        decoder : list[int], optional
            Number of channels after each decoding layer of the Unet.
        kernel_size : int or list[int], default=3
            Kernel size of the UNet.
        interpolation : int, default=1
            Interpolation order.
        grid_bound : bound_type, default='dft'
            Boundary conditions of the velocity field.
        image_bound : bound_type, default='dct2'
            Boundary conditions of the image.
        """
        super().__init__()
        self.unet = UNet(dim,
                         input_channels=2,
                         output_channels=dim,
                         encoder=encoder,
                         decoder=decoder,
                         kernel_size=kernel_size,
                         activation=tnn.LeakyReLU(0.2))
        self.exp = GridExp(interpolation=interpolation,
                           bound=grid_bound)
        self.pull = GridPull(interpolation=interpolation,
                             bound=image_bound)
        self.dim = dim
        self.image_losses = []
        self.velocity_losses = []
        self.image_metrics = {}
        self.velocity_metrics = {}

    def add_image_loss(self, *loss_fn):
        """Add one or more image loss functions.

        The image loss should measure similarity between the deformed
        source and target images.

        Parameters
        ----------
        loss_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        self.image_losses += list(loss_fn)

    def set_image_loss(self, *loss_fn):
        """Set one or more image loss functions.

        The image loss should measure similarity between the deformed
        source and target images.

        This function discards all previously held image losses.

        Parameters
        ----------
        loss_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        self.image_losses = list(loss_fn)

    def add_velocity_loss(self, *loss_fn):
        """Add one or more velocity loss functions.

        The velocity loss should penalize features of the velocity field.

        Parameters
        ----------
        loss_fn : callable or [callable, float]
            Function of one argument that returns a scalar value.

        """
        self.velocity_losses += list(loss_fn)

    def set_velocity_loss(self, *loss_fn):
        """Set one or more image loss functions.

        The velocity loss should penalize features of the velocity field.

        This function discards all previously held velocity losses.

        Parameters
        ----------
        loss_fn : callable or [callable, float]
            Function of one argument that returns a scalar value.

        """
        self.velocity_losses = list(loss_fn)

    def compute_loss(self, deformed_source, target, velocity):
        """Compute all losses."""
        loss = []
        for loss_fn in self.image_losses:
            if isinstance(loss_fn, (list, tuple)):
                _loss_fn, weight = loss_fn
                loss_fn = lambda *a, **k: weight*_loss_fn(*a, **k)
            loss.append(loss_fn(deformed_source, target))
        for loss_fn in self.velocity_losses:
            if isinstance(loss_fn, (list, tuple)):
                _loss_fn, weight = loss_fn
                loss_fn = lambda *a, **k: weight*_loss_fn(*a, **k)
            loss.append(loss_fn(velocity))
        return loss

    def add_image_metric(self, **metric_fn):
        """Add one or more image metric functions.

        The image metric should measure similarity between the deformed
        source and target images.

        Parameters
        ----------
        metric_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        self.image_metrics.update(dict(metric_fn))

    def set_image_metric(self, **metric_fn):
        """Set one or more image metric functions.

        The image metric should measure similarity between the deformed
        source and target images.

        This function discards all previously held image metrics.

        Parameters
        ----------
        metric_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        self.image_metrics = dict(metric_fn)

    def add_velocity_metric(self, **metric_fn):
        """Add one or more velocity metric functions.

        The velocity metric should penalize features of the velocity field.

        Parameters
        ----------
        metric_fn : callable or [callable, float]
            Function of one argument that returns a scalar value.

        """
        self.velocity_metrics.update(dict(metric_fn))

    def set_velocity_metric(self, **metric_fn):
        """Set one or more image metric functions.

        The velocity metric should penalize features of the velocity field.

        This function discards all previously held velocity metrics.

        Parameters
        ----------
        metric_fn : callable or [callable, float]
            Function of one argument that returns a scalar value.

        """
        self.velocity_metrics = dict(metric_fn)

    def compute_metric(self, deformed_source, target, velocity):
        """Compute all metrics."""
        metric = {}
        for key, metric_fn in self.image_metrics.items():
            key = '{}/{}/{}'.format(self.__class__.__name__, 'image', key)
            metric[key] = metric_fn(deformed_source, target)
        for key, metric_fn in self.velocity_metrics.items():
            key = '{}/{}/{}'.format(self.__class__.__name__, 'velocity', key)
            metric[key] = metric_fn(velocity)
        return metric

    def forward(self, source, target, *, _loss=None, _metric=None):
        """

        Parameters
        ----------
        source : tensor (batch, channel, *spatial)
            Source/moving image
        target : tensor (batch, channel, *spatial)
            Target/fixed image

        _loss : list, optional
            If provided, all registered losses are computed and appended.
        _metric : dict, optional
            If provided, all registered metrics are computed and appended.

        Returns
        -------
        deformed_source : tensor (batch, channel, *spatial)
            Deformed source image
        velocity : tensor (batch,, *spatial, len(spatial))
            Velocity field

        """
        # checks
        if len(source.shape) != self.dim+2:
            raise ValueError('Expected `source` to have shape (B, C, *spatial)'
                             ' with len(spatial) == {} but found {}.'
                             .format(self.dim, source.shape))
        if len(target.shape) != self.dim+2:
            raise ValueError('Expected `target` to have shape (B, C, *spatial)'
                             ' with len(spatial) == {} but found {}.'
                             .format(self.dim, target.shape))
        if not (target.shape[0] == source.shape[0] or
                target.shape[0] == 1 or source.shape[0] == 1):
            raise ValueError('Batch dimensions of `source` and `target` are '
                             'not compatible: got {} and {}'
                             .format(source.shape[0], target.shape[0]))
        if target.shape[2:] != source.shape[2:]:
            raise ValueError('Spatial dimensions of `source` and `target` are '
                             'not compatible: got {} and {}'
                             .format(source.shape[2:], target.shape[2:]))

        # chain operations
        source_and_target = torch.cat((source, target), dim=1)
        velocity = self.unet(source_and_target, _loss=_loss, _metric=_metric)
        velocity = spatial.channel2grid(velocity)
        grid = self.exp(velocity, _loss=_loss, _metric=_metric)
        deformed_source = self.pull(source, grid, _loss=_loss, _metric=_metric)

        # compute loss and metrics
        if _loss is not None:
            assert isinstance(_loss, list)
            _loss += self.compute_loss(deformed_source, target, velocity)
        if _metric is not None:
            assert isinstance(_metric, dict)
            metrics = self.compute_metric(deformed_source, target, velocity)
            self.update_metrics(_metric, metrics)

        return deformed_source, velocity
