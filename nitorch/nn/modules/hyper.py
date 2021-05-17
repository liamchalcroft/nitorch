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

        """ TODO: Add documentation
        """

        super().__init__()

        self.meta_dim = meta_dim

        if not meta_act:
            self.act = tnn.LeakyReLU()
        else:
            self.act = meta_act

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
        meta_batch = torch.split(torch.squeeze(meta), self.meta_dim)
        weight = None
        bias = None
        for meta_ in meta_batch:
            for block in self.blocks:
                meta_ = block(meta_)
                meta_ = self.act(meta_)

            weight_= self.head_w(meta_)
            bias_= self.head_b(meta_)
            if weight is None:
                weight = weight_
            else:
                torch.cat((weight,weight_))
            if bias is None:
                bias = bias_
            else:
                torch.cat((bias,bias_))

        print()

        x = F.group_norm(x, len(meta), weight=weight, bias=bias)
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
        dilation=1):

        """ TODO: Add documentation
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
        meta_batch = torch.split(torch.squeeze(meta), self.meta_dim)
        weight = None
        bias = None
        for meta_ in meta_batch:
            print(meta_.shape)
            for block in self.blocks:
                meta_ = block(meta_)
                meta_ = self.meta_act(meta_)

            weight_flat = self.head_w(meta_)
            weight_ = weight_flat.reshape(self.shape)
            if weight is None:
                weight = weight_
            else:
                torch.cat((weight,weight_), dim=1)

            if self.bias:
                bias_ = self.head_b(meta_)
                if bias is None:
                    bias = bias_
                else:
                    bias += self.head_b(meta_)

        if self.bias:
            bias /= len(meta_)

        if self.batch_norm:
            x = self.batch_norm(x, meta)

        padding = self.padding

        if padding == 'auto':
            padding = ((self.kernel_size-1)*self.dilation)//2

        if self.dim == 2:
            x = F.conv2d(x, weight, bias, 
            stride=self.stride, padding=padding, groups=len(meta))
        elif self.dim == 3:
            x = F.conv3d(x, weight, bias, 
            stride=self.stride, padding=padding, groups=len(meta))

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

        """ TODO: Add documentation
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
        meta_batch = torch.split(torch.squeeze(meta), self.meta_dim)
        weight = None
        bias = None
        for meta_ in meta_batch:
            for block in self.blocks:
                meta_ = block(meta_)
                meta_ = self.meta_act(meta_)

            weight_flat = self.head_w(meta_)
            weight_ = weight_flat.reshape(self.shape)
            if weight is None:
                weight = weight_
            else:
                torch.cat((weight,weight_), dim=1)

            if self.bias:
                bias_ = self.head_b(meta_)
                if bias is None:
                    bias = bias_
                else:
                    bias += self.head_b(meta_)

        if self.bias:
            bias /= len(meta_)

        if self.batch_norm:
            x = self.batch_norm(x, meta)

        padding = self.padding

        if padding == 'auto':
            padding = ((self.kernel_size-1)*self.dilation)//2

        if self.dim == 2:
            x = F.conv_transpose2d(x, weight, bias, 
            stride=self.stride, padding=padding, groups=len(meta))
        elif self.dim == 3:
            x = F.conv_transpose3d(x, weight, bias, 
            stride=self.stride, padding=padding, groups=len(meta))

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
            bias=False,
            residual=False):
        """

        Parameters
        ----------
        dim : {1, 2, 3}
            Number of spatial dimensions.
            
        in_channels : int
            Number of input channels.
            
        out_channels : int or sequence[int]
            Number of output channels in each convolution.
            If a sequence, multiple convolutions are performed.
            
        kernel_size : int or sequence[int], default=3
            Kernel size per dimension.
            
        activation : [sequence of] str or type or callable, default='relu'
            Activation function.
            
        batch_norm : [sequence of] bool, default=False
            Batch normalization before each convolution.
            
        stride : int or sequence[int], default=2
            Up to one value per spatial dimension.
            `output_shape \approx input_shape // stride`
            
        transposed : bool, default=False
            Make the strided convolution a transposed convolution.
            
        pool : {'max', 'min', 'median', 'mean', 'sum', None}, default=None
            Pooling used to change resolution.
            If None, the final convolution is a strided convolution.

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
                bias=b))
        else:
            modules.append(HyperConv(
                dim, i, o, meta_dim,
                kernel_size=kernel_size,
                activation=a,
                batch_norm=bn,
                stride=stride,
                bias=b))

        # print(modules)

        self.modules = modules
                
        # super().__init__(modules)

    def forward(self, x, meta, return_last=False):
        """
        """
        def is_last(layer):
            if isinstance(layer, Conv):
                if not all(s == 1 for s in make_list(layer.stride)):
                    return True
            return False

        if not isinstance(return_last, str):
            return_last = 'single' if return_last else ''

        last = []
        if 'single' in return_last:
            last.append(x[0])
        x = torch.cat(x, 1) if len(x) > 1 else x[0]
        if 'cat' in return_last:
            last.append(x)
        for layer in self.modules:
            if return_last and not is_last(layer):
                last = [x]
                if 'single' in return_last and 'cat' in return_last:
                    last = last * 2
            if residual:
                x = x + layer(x, meta)
            else:
                x = layer(x, meta)

        return (x, *last) if return_last else x
