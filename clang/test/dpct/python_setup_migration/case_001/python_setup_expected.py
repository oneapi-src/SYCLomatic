from setuptools import setup, Extension

import torch
import intel_extension_for_pytorch
SYCL_HOME = True
from intel_extension_for_pytorch.xpu import cpp_extension

from intel_extension_for_pytorch.xpu.cpp_extension import (
    
    DPCPPExtension,
    DpcppBuildExtension,
    
)
from intel_extension_for_pytorch.xpu.cpp_extension import CppExtension
