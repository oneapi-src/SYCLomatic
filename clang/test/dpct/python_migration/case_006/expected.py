from torch import xpu

devs = torch.xpu.device_count()
devs = xpu.device_count()

d_cap = torch.xpu.get_device_capability()
d_cap = xpu.get_device_capability()
d0_cap = torch.xpu.get_device_capability(devs[0])
d0_cap = xpu.get_device_capability(devs[0])

arch_list = ['']
arch_list = ['']

cuda_ver = torch.version.xpu
