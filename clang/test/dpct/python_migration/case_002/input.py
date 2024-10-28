from setuptools import setup, Extension

import torch
from torch.utils import cpp_extension

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

if CUDA_HOME:
    var = cuda_specific_op()
