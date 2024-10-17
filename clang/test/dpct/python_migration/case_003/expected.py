SYCL_HOME = __import__('os').environ.get('CMPLR_ROOT')

use_cuda = use_cuda and torch.xpu.is_available() and SYCL_HOME is not None

SYCL_HOME = __import__('os').environ.get('CMPLR_ROOT')

extension = DPCPPExtension if (use_cuda and SYCL_HOME) else CppExtension

path += __import__('os').environ.get('CMPLR_ROOT')
