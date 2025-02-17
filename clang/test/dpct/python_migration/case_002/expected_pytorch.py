from setuptools import setup, Extension

import torch
from torch.utils import cpp_extension

from torch.utils.cpp_extension import (
    CppExtension,
    CppExtension,
    BuildExtension,
    SYCL_HOME,
)

if SYCL_HOME:
    var = cuda_specific_op()
