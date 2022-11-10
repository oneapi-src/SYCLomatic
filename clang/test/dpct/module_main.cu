// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none -out-root %T/module_main %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/module_main/module_main.dp.cpp

//CHECK: #include <dlfcn.h>
#include <string>
int main(){
    //CHECK: dpct::module M;
    CUmodule M;
    //CHECK: dpct::kernel_function F;
    CUfunction F;
    std::string Path, FunctionName, Data;
    //CHECK: /*
    //CHECK-NEXT: DPCT1099:{{[0-9]+}}: User must prepare a dynamic-library for dpct::load_sycl_lib to load.  This dynamic-library should supply "wrapped" kernel-functions.
    //CHECK-NEXT: */
    //CHECK-NEXT: M = dpct::load_sycl_lib(Path.c_str());
    cuModuleLoad(&M, Path.c_str());
    //CHECK: /*
    //CHECK-NEXT: DPCT1100:{{[0-9]+}}: User must prepare a dynamic-library that the application code will copy to memory.  This dynamic-library should supply "wrapped" kernel-functions. dpct::load_sycl_lib_mem will "load" this memory for use as a library. This function creates a temporary file and can introduce a security issue.
    //CHECK-NEXT: */
    //CHECK-NEXT: M = dpct::load_sycl_lib_mem(Data.c_str());
    cuModuleLoadData(&M, Data.c_str());
    //CHECK: F = dpct::get_kernel_function(M, FunctionName.c_str());
    cuModuleGetFunction(&F, M, FunctionName.c_str());

    int sharedSize;
    CUstream s;
    void **param, **extra;
    //CHECK:  dpct::invoke_kernel_function(F, *s, sycl::range<3>(32, 16, 1), sycl::range<3>(64, 32, 4), sharedSize, param, extra);
    cuLaunchKernel(F, 1, 16, 32, 4, 32, 64, sharedSize, s, param, extra);
    //CHECK:  dpct::invoke_kernel_function(F, q_ct1, sycl::range<3>(32, 16, 1), sycl::range<3>(64, 32, 4), sharedSize, param, extra);
    cuLaunchKernel(F, 1, 16, 32, 4, 32, 64, sharedSize, 0, param, extra);
    //CHECK:  dpct::invoke_kernel_function(F, q_ct1, sycl::range<3>(32, 16, 1), sycl::range<3>(64, 32, 4), sharedSize, param, extra);
    cuLaunchKernel(F, 1, 16, 32, 4, 32, 64, sharedSize, CU_STREAM_LEGACY, param, extra);

    //CHECK: dpct::image_wrapper_base_p tex;
    //CHECK: tex = (dpct::image_wrapper_base_p)dlsym(M, "tex");
    CUtexref tex;
    cuModuleGetTexRef(&tex, M, "tex");

    //CHECK: dlclose(M);
    cuModuleUnload(M);
    return 0;
}