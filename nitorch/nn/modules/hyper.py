""" HyperNetworks (network to generate weights for second network).
Currently bare-bones but can add functionality for e.g. contextual hypernets,
Bayes-by-Hypernets, continual learning.
"""

import torch
from torch import nn as tnn
import torch.nn.functional as F
import numpy as np
from .base import nitorchmodule
from nitorch.core.py import make_list, make_tuple
from nitorch.core import utils


_native_padding_mode = ('zeros', 'reflect', 'replicate', 'circular')


# Direct copy-paste of function - import wasn't working
def expand_list(x, n, crop=False, default=None):
    """Expand ellipsis in a list by substituting it with the value
    on its left, repeated as many times as necessary. By default,
    a "virtual" ellipsis is present at the end of the list.

    expand_list([1, 2, 3],       5)            -> [1, 2, 3, 3, 3]
    expand_list([1, 2, ..., 3],  5)            -> [1, 2, 2, 2, 3]
    expand_list([1, 2, 3, 4, 5], 3, crop=True) -> [1, 2, 3]
    """
    x = list(x)
    if Ellipsis not in x:
        x.append(Ellipsis)
    idx_ellipsis = x.index(Ellipsis)
    if idx_ellipsis == 0:
        fill_value = default
    else:
        fill_value = x[idx_ellipsis-1]
    k = len(x) - 1
    x = (x[:idx_ellipsis] + 
         [fill_value] * max(0, n-k) + 
         x[idx_ellipsis+1:])
    if crop:
        x = x[:n]
    return x


@nitorchmodule
class HyperGroupNorm(tnn.Module):

    def __init__(self,
        in_channels,
        meta_dim,
        meta_depth=1,
        meta_act=None):

        """

        Group Norm layer with learnable coefficients generated by hypernet.

        Parameters
        ----------    
            
        in_channels : int or sequence[int]
            Number of channels in the input image.

        meta_dim : int
            Dimensionality of metadata input to hypernet.

        meta_depth : int, default=1
            Number of hidden layers in hypernet.

        meta_act : tnn.Module, default=None
            Optional activation function for use in hypernet.
            If None, will default to tnn.LeakyReLU().
            

        """

        super().__init__()

        self.meta_dim = meta_dim

        if not meta_act:
            self.meta_act = tnn.LeakyReLU()
        else:
            self.meta_act = meta_act
        self.final_act = tnn.Tanh()

        # define network layers
        shared_modules = []
        shared_modules.append(tnn.Linear(meta_dim, 16))
        for i in range(meta_depth):
            shared_modules.append(tnn.Linear(16*(2**i), 16*(2**(i+1)))) 

        self.blocks = tnn.ModuleList(shared_modules)

        # define output layers with shape of weights/bias

        self.head_w = tnn.Linear(16*(2**meta_depth), in_channels)
        self.head_b = tnn.Linear(16*(2**meta_depth), in_channels)

    def forward(self, x, meta):
        device = x.device

        self.meta_act = self.meta_act.to(device)
        self.final_act = self.final_act.to(device)
        self.head_w = self.head_w.to(device)
        self.head_b = self.head_b.to(device)

        for i, block in enumerate(self.blocks):
            block = block.to(device)
            meta = block(meta)
            if i < len(self.blocks)-1:
                meta = self.meta_act(meta)
            else:
                meta = self.final_act

        weight = self.head_w(meta)
        bias = self.head_b(meta)

        weight = weight.view(-1)
        bias = bias.view(-1)

        x = x.view(1, -1, *x.shape[2:])

        x = F.group_norm(x, np.prod(meta.shape[:2]), weight=weight, bias=bias)

        x = x.view(meta.shape[0], -1, *x.shape[2:])

        return x


@nitorchmodule
class HyperConv(tnn.Module):

    def __init__(self,
        dim,
        in_channels,
        out_channels,
        meta_dim,
        activation=True,
        batch_norm=True,
        bias=True,
        kernel_size=3,
        meta_depth=1,
        meta_act=None,
        stride=1,
        padding='auto',
        padding_mode='zeros',
        output_padding=0,
        dilation=1,
        grouppool=False):

        """

        Convolution layer with weights generated by hypernet.

        Parameters
        ----------   

        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : int or sequence[int]
            Number of channels in the input image.
            If a sequence, grouped convolutions are used.
            
        out_channels : int or sequence[int]
            Number of channels produced by the convolution.
            If a sequence, grouped convolutions are used.

        meta_dim : int
            Dimensionality of metadata input to hypernet.

        activation : str or type or callable, optional
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation

        batch_norm : bool or type or callable, optional
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).
            If True, will default to HyperGroupNorm.

        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
            For multiple inputs, mean bias calculated.
            
        kernel_size : int or sequence[int]
            Size of the convolution kernel.

        meta_depth : int, default=1
            Number of hidden layers in hypernet.

        meta_act : tnn.Module, default=None
            Optional activation function for use in hypernet.
            If None, will default to tnn.LeakyReLU().
            
        stride : int or sequence[int], default=1:
            Stride of the convolution.
            
        padding : int or sequence[int] or 'auto', default=0
            Zero-padding added to all three sides of the input.
            
        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.
            
        output_padding : int or sequence[int], default=0
            Additional size added to (the bottom/right) side of each
            dimension in the output shape. Only used if `transposed is True`.
            
        dilation : int or sequence[int], default=1
            Spacing between kernel elements.

        grouppool : bool, default=False
            Optional to activate 'group-pooling' behaviour.
            If True, will perform convolution with stack of all modality weights.

        """

        super().__init__()

        self.meta_dim = meta_dim
        self.dim = dim
        self.stride = stride
        self.bias = bias
        self.padding = padding
        self.batch_norm = batch_norm
        self.activation = activation
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.grouppool = grouppool

        if batch_norm == True:
            self.batch_norm = HyperGroupNorm(in_channels, meta_dim, meta_depth, meta_act)

        if activation == True:
            self.activation = tnn.LeakyReLU()

        if not meta_act:
            self.meta_act = tnn.LeakyReLU()
        else:
            self.meta_act = meta_act
        self.final_act = tnn.Tanh()

        # define network layers
        shared_modules = []
        shared_modules.append(tnn.Linear(meta_dim, 16))
        for i in range(meta_depth):
            shared_modules.append(tnn.Linear(16*(2**i), 16*(2**(i+1)))) 

        self.blocks = tnn.ModuleList(shared_modules)

        # define output layers with shape of weights/bias
        if dim == 2:
            self.shape = [out_channels, in_channels, kernel_size, kernel_size]
        elif dim == 3:
            self.shape = [out_channels, in_channels, kernel_size, kernel_size, kernel_size]

        self.head_w = tnn.Linear(16*(2**meta_depth), np.prod(self.shape))
        if bias:
            self.head_b = tnn.Linear(16*(2**meta_depth), out_channels)

        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding = output_padding

    def forward(self, x, meta):

        device = x.device

        self.meta_act = self.meta_act.to(device)
        self.head_w = self.head_w.to(device)
        if self.bias:
            self.head_b = self.head_b.to(device)

        if self.batch_norm:
            x = self.batch_norm(x, meta)

        padding = self.padding

        if padding == 'auto':
            padding = ((self.kernel_size-1)*self.dilation)//2

        shape = self.shape.copy()
        shape[0] *= np.prod(meta.shape[:2])
        
        for i, block in enumerate(self.blocks):
            block = block.to(device)
            meta = block(meta)
            if i < len(self.blocks)-1:
                meta = self.meta_act(meta)
            else:
                meta = self.final_act

        weight = self.head_w(meta)

        if self.grouppool==True:
            # weight = torch.mean(weight, dim=1)
            shape[0] //= meta.shape[1]
            shape[1] *= meta.shape[1]
            
        weight = weight.view(shape)

        if self.bias:
            bias = self.head_b(meta)
            if self.grouppool==True:
                bias = torch.mean(bias, dim=1)
            bias = bias.view(-1)
            bias = bias.to(device)

        weight = weight.to(device)

        x = x.view(1, -1, *x.shape[2:])

        if self.dim == 2:
            if self.grouppool==True:
                x = F.conv2d(x, weight, bias, 
                # stride=self.stride, padding=padding, groups=np.prod(meta.shape[:2]))
                # stride=self.stride, padding=padding)
                stride=self.stride, padding=padding, groups=meta.shape[0])
            else:
                x = F.conv2d(x, weight, bias, 
                stride=self.stride, padding=padding, groups=np.prod(meta.shape[:2]))

        elif self.dim == 3:
            if self.grouppool==True:
                x = F.conv3d(x, weight, bias, 
                # stride=self.stride, padding=padding, groups=np.prod(meta.shape[:2]))
                # stride=self.stride, padding=padding)
                stride=self.stride, padding=padding, groups=meta.shape[0])
            else:    
                x = F.conv3d(x, weight, bias, 
                stride=self.stride, padding=padding, groups=np.prod(meta.shape[:2]))

        x = x.view(meta.shape[0], -1, *x.shape[2:])

        # perform post-padding
        if self.output_padding:
            x = utils.pad(x, self.output_padding, side='right')

        if self.activation:
            x = self.activation(x)

        return x


@nitorchmodule
class HyperConvTranspose(tnn.Module):

    def __init__(self,
        dim,
        in_channels,
        out_channels,
        meta_dim,
        batch_norm=True,
        activation=True,
        bias=True,
        kernel_size=3,
        meta_depth=1,
        meta_act=None,
        stride=1,
        padding='auto',
        padding_mode='zeros',
        output_padding=0,
        dilation=1):

        """

        Transposed convolution layer with weights generated by hypernet.

        Parameters
        ----------   

        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : int or sequence[int]
            Number of channels in the input image.
            If a sequence, grouped convolutions are used.
            
        out_channels : int or sequence[int]
            Number of channels produced by the convolution.
            If a sequence, grouped convolutions are used.

        meta_dim : int
            Dimensionality of metadata input to hypernet.

        activation : str or type or callable, optional
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation

        batch_norm : bool or type or callable, optional
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).
            If True, will default to HyperGroupNorm.

        bias : bool, default=True
            If ``True``, adds a learnable bias to the output.
            For multiple inputs, mean bias calculated.
            
        kernel_size : int or sequence[int]
            Size of the convolution kernel.

        meta_depth : int, default=1
            Number of hidden layers in hypernet.

        meta_act : tnn.Module, default=None
            Optional activation function for use in hypernet.
            If None, will default to tnn.LeakyReLU().
            
        stride : int or sequence[int], default=1:
            Stride of the convolution.
            
        padding : int or sequence[int] or 'auto', default=0
            Zero-padding added to all three sides of the input.
            
        padding_mode : {'zeros', 'reflect', 'replicate', 'circular'}, default='zeros'
            Padding mode.
            
        output_padding : int or sequence[int], default=0
            Additional size added to (the bottom/right) side of each
            dimension in the output shape. Only used if `transposed is True`.
            
        dilation : int or sequence[int], default=1
            Spacing between kernel elements.

        grouppool : bool, default=False
            Optional to activate 'group-pooling' behaviour.
            If True, will perform convolution with stack of all modality weights.

        """

        super().__init__()

        self.meta_dim = meta_dim
        self.dim = dim
        self.stride = stride
        self.bias = bias
        self.padding = padding
        self.batch_norm = batch_norm
        self.activation = activation
        self.kernel_size = kernel_size
        self.dilation = dilation

        if batch_norm == True:
            self.batch_norm = HyperGroupNorm(in_channels, meta_dim, meta_depth, meta_act)

        if activation == True:
            self.activation = tnn.LeakyReLU()

        if not meta_act:
            self.meta_act = tnn.LeakyReLU()
        else:
            self.meta_act = meta_act
        self.final_act = tnn.Tanh()

        # define network layers
        shared_modules = []
        shared_modules.append(tnn.Linear(meta_dim, 16))
        for i in range(meta_depth):
            shared_modules.append(tnn.Linear(16*(2**i), 16*(2**(i+1)))) 

        self.blocks = tnn.ModuleList(shared_modules)

        # define output layers with shape of weights/bias
        if dim == 2:
            self.shape = [in_channels, out_channels, kernel_size, kernel_size]
        elif dim == 3:
            self.shape = [in_channels, out_channels, kernel_size, kernel_size, kernel_size]

        self.head_w = tnn.Linear(16*(2**meta_depth), np.prod(self.shape))
        if bias:
            self.head_b = tnn.Linear(16*(2**meta_depth), out_channels)

        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding = output_padding

    def forward(self, x, meta):

        device = x.device

        self.meta_act = self.meta_act.to(device)
        self.head_w = self.head_w.to(device)
        if self.bias:
            self.head_b = self.head_b.to(device)

        if self.batch_norm:
            x = self.batch_norm(x, meta)

        padding = self.padding

        if padding == 'auto':
            padding = ((self.kernel_size-1)*self.dilation)//2

        shape = self.shape.copy()
        shape[1] *= np.prod(meta.shape[:2])
        
        for i, block in enumerate(self.blocks):
            block = block.to(device)
            meta = block(meta)
            if i < len(self.blocks)-1:
                meta = self.meta_act(meta)
            else:
                meta = self.final_act

        weight = self.head_w(meta)

        if self.grouppool==True:
            weight = torch.mean(weight, dim=1)
            shape[1] //= meta.shape[1]
            
        weight = weight.view(shape)

        if self.bias:
            bias = self.head_b(meta)
            if self.grouppool==True:
                bias = torch.mean(bias, dim=1)
            bias = bias.view(-1)
            bias = bias.to(device)

        weight = weight.to(device)

        x = x.view(1, -1, *x.shape[2:])

        if self.dim == 2:
            if self.grouppool==True:
                x = F.conv_transpose2d(x, weight, bias, 
                stride=self.stride, padding=padding, groups=np.prod(meta.shape[:2]))
            else:
                x = F.conv_transpose2d(x, weight, bias, 
                stride=self.stride, padding=padding, groups=np.prod(meta.shape[:2]))

        elif self.dim == 3:
            if self.grouppool==True:
                x = F.conv_transpose3d(x, weight, bias, 
                stride=self.stride, padding=padding, groups=np.prod(meta.shape[:2]))
            else:    
                x = F.conv_transpose3d(x, weight, bias, 
                stride=self.stride, padding=padding, groups=np.prod(meta.shape[:2]))

        x = x.view(meta.shape[0], -1, *x.shape[2:])

        # perform post-padding
        if self.output_padding:
            x = utils.pad(x, self.output_padding, side='right')

        if self.activation:
            x = self.activation(x)

        return x
        

@nitorchmodule
class HyperStack(tnn.Module):

    def __init__(
            self,
            dim,
            in_channels,
            out_channels,
            meta_dim,
            kernel_size=3,
            stride=1,
            meta_depth=1,
            meta_act=None,
            transposed=False,
            pool=None,
            activation=True,
            batch_norm=True,
            bias=True,
            residual=False,
            return_last=False,
            grouppool=False):
        
        """

        Stack of convolution layers with weights generated by hypernet.

        Parameters
        ----------   

        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : int or sequence[int]
            Number of channels in the input image.
            If a sequence, grouped convolutions are used.
            
        out_channels : int or sequence[int]
            Number of channels produced by the convolution.
            If a sequence, grouped convolutions are used.

        meta_dim : int
            Dimensionality of metadata input to hypernet.
            
        kernel_size : int or sequence[int]
            Size of the convolution kernel.

        stride : int or sequence[int], default=1:
            Stride of the convolution.

        meta_depth : int, default=1
            Number of hidden layers in hypernet.

        meta_act : tnn.Module, default=None
            Optional activation function for use in hypernet.
            If None, will default to tnn.LeakyReLU().

        transposed : bool, default=False
            Make the strided convolution a transposed convolution.

        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Pooling used to change resolution.
            If None, the final convolution is a strided convolution.

        activation : str or type or callable, optional
            Activation function. An activation can be a class
            (typically a Module), which is then instantiated, or a
            callable (an already instantiated class or a more simple
            function). It is useful to accept both these cases as they
            allow to either:
                * have a learnable activation specific to this module
                * have a learnable activation shared with other modules
                * have a non-learnable activation

        batch_norm : bool or type or callable, optional
            Batch normalization layer.
            Can be a class (typically a Module), which is then instantiated,
            or a callable (an already instantiated class or a more simple
            function).
            If True, will default to HyperGroupNorm.

        bias : [sequence of] int, default=True
            Include a bias term in the convolution.

        residual : bool, default=False
            Add residual connections between convolutions.
            This has no effect if only one convolution is performed.
            No residual connection is applied to the output of the last
            layer (strided conv or pool).

        return_last : {'single', 'cat', 'single+cat'} or bool, default=False
            Return the last output before up/downsampling on top of the
            real output (useful for skip connections).

            'single' and 'cat' are useful when the stacked convolution contains
            a single strided convolution and takes as input a skipped
            connection. 'single' only returns the first input argument
            whereas 'cat' returns all concatenated input arguments.
            `True` is equivalent to 'single'.

        grouppool : bool, default=False
            Optional to activate 'group-pooling' behaviour.
            If True, final convolution will use stack of all modality weights.

        """

        super().__init__()

        self.meta_dim = meta_dim
        self.dim = dim
        self.residual = residual

        out_channels = make_list(out_channels)
        in_channels = [in_channels] + out_channels[:-1]
        nb_layers = len(out_channels)
        
        activation = expand_list(make_list(activation), nb_layers, default='relu')
        batch_norm = expand_list(make_list(batch_norm), nb_layers, default=True)
        bias = expand_list(make_list(bias), nb_layers, default=True)
        
        if pool and transposed:
            raise ValueError('Cannot have both `pool` and `transposed`.')
      
        all_shapes = zip(
            in_channels, 
            out_channels,
            activation,
            batch_norm,
            bias)
        *all_shapes, final_shape = all_shapes
        
        # stacked conv (without strides)
        modules = []
        for d, (i, o, a, bn, b) in enumerate(all_shapes):
            modules.append(HyperConv(
                dim, i, o, meta_dim,
                kernel_size=kernel_size,
                activation=a,
                batch_norm=bn
                ))
        
        # last conv (strided if not pool)
        i, o, a, bn, b = final_shape
        if transposed:
            modules.append(HyperConvTranspose(
                dim, i, o, meta_dim,
                kernel_size=kernel_size,
                activation=a,
                batch_norm=bn,
                stride=stride,
                bias=b,
                grouppool=grouppool))
        else:
            modules.append(HyperConv(
                dim, i, o, meta_dim,
                kernel_size=kernel_size,
                activation=a,
                batch_norm=bn,
                stride=stride,
                bias=b,
                grouppool=grouppool))

        self.modules = modules
                
    def forward(self, x, meta, return_last=False):
        def is_last(layer):
            if isinstance(layer, HyperConv):
                if not all(s == 1 for s in make_list(layer.stride)):
                    return True
            return False

        if not isinstance(return_last, str):
            return_last = 'single' if return_last else ''

        if return_last:
            last = x
        for layer in self.modules:
            if return_last and not is_last(layer):
                last = x
            if self.residual:
                x = x + layer(x, meta)
            else:
                x = layer(x, meta)

        return (x, last) if return_last else x
