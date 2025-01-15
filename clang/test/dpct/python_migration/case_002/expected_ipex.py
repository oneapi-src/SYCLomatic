from setuptools import setup, Extension

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch.xpu import cpp_extension

from intel_extension_for_pytorch.xpu.cpp_extension import (
    
    DPCPPExtension,
    DpcppBuildExtension,
    
)
from intel_extension_for_pytorch.xpu.cpp_extension import CppExtension
SYCL_HOME = __import__('os').environ.get('CMPLR_ROOT')

if SYCL_HOME:
    var = cuda_specific_op()
