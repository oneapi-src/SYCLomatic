from torch.utils.cpp_extension import SYCL_HOME

use_cuda = use_cuda and torch.xpu.is_available() and SYCL_HOME is not None

from torch.utils.cpp_extension import (SYCL_HOME)

extension = CppExtension if (use_cuda and SYCL_HOME) else CppExtension

path += torch.utils.cpp_extension.SYCL_HOME
