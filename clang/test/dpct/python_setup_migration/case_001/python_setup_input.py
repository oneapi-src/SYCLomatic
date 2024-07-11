from setuptools import setup, Extension

import torch
from torch.utils import cpp_extension

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
extension = CUDAExtension if use_cuda else CppExtension

cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', ['quant_cuda.cpp', 'quant_cuda_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
