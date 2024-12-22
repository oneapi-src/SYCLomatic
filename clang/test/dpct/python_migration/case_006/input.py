from torch import cuda

cuda_ver = torch.version.cuda

#init
torch.cuda.init()
cuda.init()
is_init = torch.cuda.is_initialized()
is_init = cuda.is_initialized()

# device APIs
devs = torch.cuda.device_count()
devs = cuda.device_count()

dev = torch.cuda.current_device()
dev = cuda.current_device()

torch.cuda.set_device(dev)
cuda.set_device(dev)

d_props = torch.cuda.get_device_properties(dev)
d_props = cuda.get_device_properties(dev)

curr_d_name = torch.cuda.get_device_name()
curr_d_name = cuda.get_device_name()
d_name = torch.cuda.get_device_name(dev)
d_name = cuda.get_device_name(dev)

d_cap = torch.cuda.get_device_capability()
d_cap = cuda.get_device_capability()
d0_cap = torch.cuda.get_device_capability(devs[0])
d0_cap = cuda.get_device_capability(devs[0])

dev_of_obj = torch.cuda.device_of(obj)
dev_of_obj = cuda.device_of(obj)

arch_list = torch.cuda.get_arch_list()
arch_list = cuda.get_arch_list()

torch.cuda.synchronize()
cuda.synchronize()
torch.cuda.synchronize(dev)
cuda.synchronize(dev)

# stream APIs
curr_st = torch.cuda.current_stream()
curr_st = cuda.current_stream()
curr_d_st = torch.cuda.current_stream(dev)
curr_d_st = cuda.current_stream(dev)

st = torch.cuda.StreamContext(curr_st)
st = cuda.StreamContext(curr_st)

stS = torch.cuda.stream(st)
stS = cuda.stream(st)

torch.cuda.set_stream(st)
cuda.set_stream(st)
