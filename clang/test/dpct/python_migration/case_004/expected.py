cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.dp.cpp")))

# srcs in normal args (single & double quotes)
## single quote
out = func('foo.cpp', 'bar.dp.cpp')
## double quotes
out = func("foo.cpp", "bar.dp.cpp")

# srcs in list arg
## single quote
out = func(['foo.cpp', 'bar.dp.cpp'])
## double quotes
out = func(["foo.cpp", "bar.dp.cpp"])
