setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.DPCPPExtension(
        'quant_cuda', ["quant_cuda.cpp", 'quant_cuda_kernel.dp.cpp']
    , include_dirs=cpp_extension.include_paths(),)],
    cmdclass={'build_ext': cpp_extension.DpcppBuildExtension}
)
