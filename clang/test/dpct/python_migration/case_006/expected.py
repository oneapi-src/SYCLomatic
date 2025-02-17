from torch import xpu

cuda_ver = torch.version.xpu

#init
torch.xpu.init()
xpu.init()
is_init = torch.xpu.is_initialized()
is_init = xpu.is_initialized()

# device APIs
devs = torch.xpu.device_count()
devs = xpu.device_count()

dev = torch.xpu.current_device()
dev = xpu.current_device()

torch.xpu.set_device(dev)
xpu.set_device(dev)

d_props = torch.xpu.get_device_properties(dev)
d_props = xpu.get_device_properties(dev)

curr_d_name = torch.xpu.get_device_name()
curr_d_name = xpu.get_device_name()
d_name = torch.xpu.get_device_name(dev)
d_name = xpu.get_device_name(dev)

d_cap = torch.xpu.get_device_capability()
d_cap = xpu.get_device_capability()
d0_cap = torch.xpu.get_device_capability(devs[0])
d0_cap = xpu.get_device_capability(devs[0])

dev_of_obj = torch.xpu.device_of(obj)
dev_of_obj = xpu.device_of(obj)

arch_list = ['']
arch_list = ['']

torch.xpu.synchronize()
xpu.synchronize()
torch.xpu.synchronize(dev)
xpu.synchronize(dev)

# stream APIs
curr_st = torch.xpu.current_stream()
curr_st = xpu.current_stream()
curr_d_st = torch.xpu.current_stream(dev)
curr_d_st = xpu.current_stream(dev)

st = torch.xpu.StreamContext(curr_st)
st = xpu.StreamContext(curr_st)

stS = torch.xpu.stream(st)
stS = xpu.stream(st)

torch.xpu.set_stream(st)
xpu.set_stream(st)
