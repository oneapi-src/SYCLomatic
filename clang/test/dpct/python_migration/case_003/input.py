from torch.utils.cpp_extension import CUDA_HOME

use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None

from torch.utils.cpp_extension import (CUDA_HOME)

extension = CUDAExtension if (use_cuda and CUDA_HOME) else CppExtension

path += torch.utils.cpp_extension.CUDA_HOME
