from setuptools import setup, Extension

import torch
import intel_extension_for_pytorch
import os
SYCL_HOME = os.environ.get('CMPLR_ROOT')
from intel_extension_for_pytorch.xpu import cpp_extension

from intel_extension_for_pytorch.xpu.cpp_extension import (
    
    DPCPPExtension,
    DpcppBuildExtension,
    
)
from intel_extension_for_pytorch.xpu.cpp_extension import CppExtension
