// RUN: dpct --format-range=none -out-root %T/module_main %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/module_main/module_main.dp.cpp

//CHECK: #include <dpct/dpct.hpp>

#include "cuda.h"
#include <string>
int main(){
    //CHECK: dpct::kernel_library M;
    CUmodule M;
    //CHECK: dpct::kernel_function F;
    CUfunction F;
    std::string Path, FunctionName, Data;
    //CHECK: /*
    //CHECK-NEXT: DPCT1103:{{[0-9]+}}: 'Path.c_str()' should be a dynamic library. The dynamic library should supply wrapped kernel functions.
    //CHECK-NEXT: */
    //CHECK-NEXT: M = dpct::load_kernel_library(Path.c_str());
    cuModuleLoad(&M, Path.c_str());
    //CHECK: /*
    //CHECK-NEXT: DPCT1104:{{[0-9]+}}: 'Data.c_str()' should point to a dynamic library loaded in memory. The dynamic library should supply wrapped kernel functions.
    //CHECK-NEXT: */
    //CHECK-NEXT: M = dpct::load_kernel_library_mem(Data.c_str());
    cuModuleLoadData(&M, Data.c_str());

    //CHECK: /*
    //CHECK-NEXT: DPCT1104:{{[0-9]+}}: 'Data.c_str()' should point to a dynamic library loaded in memory. The dynamic library should supply wrapped kernel functions.
    //CHECK-NEXT: */
    //CHECK-NEXT: M = dpct::load_kernel_library_mem(Data.c_str());
    cuModuleLoadDataEx(&M, Data.c_str(), 0, NULL, NULL);

    //CHECK: F = dpct::get_kernel_function(M, FunctionName.c_str());
    cuModuleGetFunction(&F, M, FunctionName.c_str());


    int    *argBuffer[3];
    size_t  argBufferSize = sizeof(argBuffer);
    //CHECK: void *extra[] = {((void *) 2), &argBufferSize,
    //CHECK-NEXT: ((void *) 1), argBuffer,
    //CHECK-NEXT: ((void *) 0)};
    void   *extra[] = {CU_LAUNCH_PARAM_BUFFER_SIZE, &argBufferSize,
                       CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
                       CU_LAUNCH_PARAM_END};

    int sharedSize;
    CUstream s;
    void **param;
    //CHECK:  dpct::invoke_kernel_function(F, *s, sycl::range<3>(32, 16, 1), sycl::range<3>(64, 32, 4), sharedSize, param, extra);
    cuLaunchKernel(F, 1, 16, 32, 4, 32, 64, sharedSize, s, param, extra);
    //CHECK:  dpct::invoke_kernel_function(F, q_ct1, sycl::range<3>(32, 16, 1), sycl::range<3>(64, 32, 4), sharedSize, param, extra);
    cuLaunchKernel(F, 1, 16, 32, 4, 32, 64, sharedSize, 0, param, extra);
    //CHECK:  dpct::invoke_kernel_function(F, q_ct1, sycl::range<3>(32, 16, 1), sycl::range<3>(64, 32, 4), sharedSize, param, extra);
    cuLaunchKernel(F, 1, 16, 32, 4, 32, 64, sharedSize, CU_STREAM_LEGACY, param, extra);

    //CHECK: dpct::image_wrapper_base_p tex;
    //CHECK: tex = dpct::get_image_wrapper(M, "tex");
    CUtexref tex;
    cuModuleGetTexRef(&tex, M, "tex");

    //CHECK: dpct::unload_kernel_library(M);
    cuModuleUnload(M);

    //CHECK: if (DPCT_CHECK_ERROR(dpct::unload_kernel_library(M))==0) {
    if (cuModuleUnload(M)==0) {
      printf("unload failed\n");
    }

    return 0;
}