cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

# srcs in normal args (single & double quotes)
## single quote
out = func('foo.cpp', 'bar.cu')
## double quotes
out = func("foo.cpp", "bar.cu")

# srcs in list arg
## single quote
out = func(['foo.cpp', 'bar.cu'])
## double quotes
out = func(["foo.cpp", "bar.cu"])
