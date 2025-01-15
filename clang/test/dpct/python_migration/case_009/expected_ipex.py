import torch
import intel_extension_for_pytorch

intel_extension_for_pytorch.xpu.cpp_extension.load(name=module_name, build_directory=cached_build_dir,
                verbose=verbose_build, sources=cached_sources, **build_kwargs, extra_cflags=['-fsycl'], extra_ldflags=['-fsycl'])
