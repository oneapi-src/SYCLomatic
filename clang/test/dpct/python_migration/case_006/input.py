from torch import cuda

devs = torch.cuda.device_count()
devs = cuda.device_count()

d_cap = torch.cuda.get_device_capability()
d_cap = cuda.get_device_capability()
d0_cap = torch.cuda.get_device_capability(devs[0])
d0_cap = cuda.get_device_capability(devs[0])

arch_list = torch.cuda.get_arch_list()
arch_list = cuda.get_arch_list()

cuda_ver = torch.version.cuda
