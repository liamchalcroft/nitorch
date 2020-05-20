#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:26:01 2020

@author: ybalba
"""

# ~~~ imports
import os
import sys
import re
import subprocess
import glob
import packaging.version
import collections
import distutils
from setuptools import setup, command, find_packages, Extension
from buildtools import *
import torch


# ~~~ libnitorch files
# Note that sources are identical between libnitorch_cpu and libnitorch_cuda.
# The same code is compiled in two different ways to generate native and cuda
# code. This trick allows to minimize code duplication.
# Finally, libnitorch links to both these sub-libraries and dispatches
# according to the Tensor's device type.
libnitorch_cpu_sources = ['nitorch/_C/pushpull_common.cpp']
libnitorch_cuda_sources = ['nitorch/_C/pushpull_common.cpp']
libnitorch_sources = ['nitorch/_C/pushpull.cpp']

# TODO
# . There is still quite a lot to do in setup and buildtools in order to make
#   things clean and work on multiple platforms.
# . I have to add abi checks and other smart tricks as in
#   torch.utils.cpp_extension

MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)

# ~~~ helpers
# Most of the helpers are in build tools. The remainign helpers defined here
# are specific to the version of pytorch that we compile against.

def torch_version():
    return packaging.version.parse(torch.__version__).release()


def torch_cuda_version():
    version = torch._C._cuda_getCompiledVersion()
    version = (version//1000, version//10 % 1000, version % 10)
    return version


def torch_cudnn_version():
    return torch._C._cudnn.getCompileVersion()


def torch_parallel_backend():
    match = re.search('^ATen parallel backend: (?P<backend>.*)$',
                      torch._C._parallel_info(), re.MULTILINE)
    if match is None:
        return None
    backend = match.group('backend')
    if backend == 'OpenMP':
        return 'AT_PARALLEL_OPENMP'
    elif backend == 'native thread pool':
        return 'AT_PARALLEL_NATIVE'
    elif backend == 'native thread pool and TBB':
        return 'AT_PARALLEL_NATIVE_TBB'
    else:
        return None


def torch_abi():
  return str(int(torch._C._GLIBCXX_USE_CXX11_ABI))


def torch_libraries(use_cuda=False):
    libraries = ['c10', 'torch_cpu', 'torch_python']
    if use_cuda:
        libraries += ['cudart', 'c10_cuda', 'torch_cuda']
    return libraries


def torch_library_dirs(use_cuda=False, use_cudnn=False):
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_library_dir = os.path.join(torch_dir, 'lib')
    library_dirs = [torch_library_dir]
    if use_cuda:
        if is_windows():
            library_dirs += [os.path.join(cuda_home(), 'lib/x64')]
        elif os.path.exists(os.path.join(cuda_home(), 'lib64')):
            library_dirs += [os.path.join(cuda_home(), 'lib64')]
        elif os.path.exists(os.path.join(cuda_home(), 'lib')):
            library_dirs += [os.path.join(cuda_home(), 'lib')]
    if use_cudnn:
        if is_windows():
            library_dirs += [os.path.join(cudnn_home(), 'lib/x64')]
        elif os.path.exists(os.path.join(cudnn_home(), 'lib64')):
            library_dirs += [os.path.join(cudnn_home(), 'lib64')]
        elif os.path.exists(os.path.join(cudnn_home(), 'lib')):
            library_dirs += [os.path.join(cudnn_home(), 'lib')]
    return library_dirs


def torch_include_dirs(use_cuda=False, use_cudnn=False):
    torch_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_include_dir = os.path.join(torch_dir, 'include')
    include_dirs = [torch_include_dir,
                    os.path.join(torch_include_dir, 'torch', 'csrc', 'api', 'include'),
                    os.path.join(torch_include_dir, 'TH'),
                    os.path.join(torch_include_dir, 'THC')]
    if use_cuda:
        cuda_include_dir = os.path.join(cuda_home(), 'include')
        if cuda_include_dir != '/usr/include':
            include_dirs += [cuda_include_dir]
    if use_cudnn:
        include_dirs += [os.path.join(cudnn_home(), 'include')]
    return include_dirs


def cuda_check():
    local_version = cuda_version()
    torch_version = torch_cuda_version()
    ok = (local_version[0] == torch_version[0] and
          local_version[1] == torch_version[1])
    if not ok:
        print('Your version of CUDA is v{}.{} while PyTorch was compiled with'
              'CUDA v{}.{}. NiTorch cannot be compiled with CUDA.'.format(
              local_version[0], local_version[1],
              torch_version[0], torch_version[1]))
    return ok


def cudnn_check():
    local_version = cudnn_version()
    torch_version = torch_cudnn_version()
    ok = (local_version[0] == torch_version[0] and
          local_version[1] == torch_version[1])
    if not ok:
        print('Your version of CuDNN is v{}.{} while PyTorch was compiled with'
              'CuDNN v{}.{}. NiTorch cannot be compiled with CuDNN.'.format(
              local_version[0], local_version[1],
              torch_version[0], torch_version[1]))
    return ok


def cuda_arch_flags():
    """
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """

    # Note: keep combined names ("arch1+arch2") above single names, otherwise
    # string replacement may not do the right thing
    named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5']
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

    # The default is sm_30 for CUDA 9.x and 10.x
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)

    # If not given, determine what's needed for the GPU that can be found
    if not arch_list:
        capability = torch.cuda.get_device_capability()
        arch_list = ['{}.{}'.format(capability[0], capability[1])]
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        arch_list = arch_list.replace(' ', ';')
        # Expand named arches
        for named_arch, archval in named_arches.items():
            arch_list = arch_list.replace(named_arch, archval)

        arch_list = arch_list.split(';')

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError("Unknown CUDA arch ({}) or GPU not supported".format(arch))
        else:
            num = arch[0] + arch[2]
            flags.append('-gencode=arch=compute_{},code=sm_{}'.format(num, num))
            if arch.endswith('+PTX'):
                flags.append('-gencode=arch=compute_{},code=compute_{}'.format(num, num))

    return list(set(flags))


def torch_extension_flags(name):
    return ['-DTORCH_EXTENSION_NAME={}'.format(name),
            '-DTORCH_API_INCLUDE_EXTENSION_H']


def gcc_clang_flags():
    return ['-fPIC', '-std=c++14']


def msvc_flags():
    return ['/MD', '/wd4819', '/EHsc']


def nvcc_flags():
    return [
      '-x=cu',
      '-D__CUDA_NO_HALF_OPERATORS__',
      '-D__CUDA_NO_HALF_CONVERSIONS__',
      '-D__CUDA_NO_HALF2_OPERATORS__',
      '--expt-relaxed-constexpr']


def omp_flags():
    if is_windows():
        return ['/openmp']
    elif is_darwin():
        # https://stackoverflow.com/questions/37362414/
        return ['-fopenmp=libiomp5']
    else:
        return ['-fopenmp']


def common_flags():
    if is_windows():
        return msvc_flags()
    else:
        return gcc_clang_flags()


def cuda_flags():
    flags = nvcc_flags() + cuda_arch_flags()
    if is_windows():
        for flag in common_flags():
            flags = ['-Xcompiler', flag] + flags
    else:
        for flag in common_flags():
            flags += ['--compiler-options', flag]
    return flags

# ~~~ checks
use_cuda = cuda_home() and cuda_check()
use_cudnn = cudnn_home() and cudnn_check()
use_omp = os.environ.get('USE_OPENMP', True)

# ~~~ setup libraries
build_libraries = []

libnitorch_cpu = ('nitorch_cpu', {
  'sources': libnitorch_cpu_sources,
  'include_dirs': torch_include_dirs(),
  'define': {torch_parallel_backend(): '1',
             '_GLIBCXX_USE_CXX11_ABI': torch_abi()},
  'extra_compile_args': common_flags() + (omp_flags() if use_omp else []),
  'language': 'c++',
})
build_libraries += [libnitorch_cpu]

if use_cuda or use_cudnn:
    libnitorch_cuda = ('nitorch_cuda', {
      'sources': libnitorch_cuda_sources,
      'include_dirs': torch_include_dirs(use_cuda, use_cudnn),
      'define': {torch_parallel_backend(): '1',
                 '_GLIBCXX_USE_CXX11_ABI': torch_abi()},
      'extra_compile_args': cuda_flags(),
      'language': 'cuda',
    })
    build_libraries += [libnitorch_cuda]

libnitorch = ('nitorch', {
  'sources': libnitorch_sources,
  'include_dirs': torch_include_dirs(),
  'define': {torch_parallel_backend(): '1',
             '_GLIBCXX_USE_CXX11_ABI': torch_abi()},
  'extra_compile_args': common_flags() + (['-DNI_WITH_CUDA'] if use_cuda else []),
  'language': 'c++',
})
build_libraries += [libnitorch]


# ~~~ setup extensions
nitorch_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nitorch', 'lib')
build_extensions = []
SpatialExtension = Extension(
    name='_C.spatial',
    sources=['nitorch/_C/spatial.cpp'],
    libraries=torch_libraries(use_cuda),
    library_dirs=torch_library_dirs(use_cuda, use_cudnn) + [nitorch_lib_path],
    include_dirs=torch_include_dirs(use_cuda, use_cudnn),
    extra_compile_args=common_flags() + torch_extension_flags('spatial'),
    runtime_library_dirs=[link_relative(os.path.join('..', 'lib'))]
)
build_extensions += [SpatialExtension]


setup(
    name='nitorch',
    version='0.1a.dev',
    packages=find_packages(),
    install_requires=['torch>=1.5'],
    python_requires='>=3.0',
    setup_requires=['torch>=1.5'],
    ext_package='nitorch',
    libraries=build_libraries,
    ext_modules=build_extensions,
    cmdclass={'build_clib': build_shared_clib}
)