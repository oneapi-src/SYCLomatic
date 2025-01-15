import torch

torch.utils.cpp_extension.load(name=module_name, build_directory=cached_build_dir,
                verbose=verbose_build, sources=cached_sources, **build_kwargs, extra_cflags=['-fsycl'], extra_ldflags=['-fsycl'])
