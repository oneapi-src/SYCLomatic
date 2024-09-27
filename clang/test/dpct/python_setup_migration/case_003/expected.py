use_cuda = use_cuda and torch.xpu.is_available() and SYCL_HOME is not None
extension = DPCPPExtension if use_cuda else CppExtension
