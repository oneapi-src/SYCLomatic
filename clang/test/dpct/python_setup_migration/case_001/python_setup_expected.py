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

use_cuda = use_cuda and torch.xpu.is_available() and SYCL_HOME is not None
extension = DPCPPExtension if use_cuda else CppExtension

cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cpp")))

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.DPCPPExtension(
        'quant_cuda', ['quant_cuda.cpp', 'quant_cuda_kernel.cpp']
    , include_dirs=cpp_extension.include_paths(),)],
    cmdclass={'build_ext': cpp_extension.DpcppBuildExtension}
)
