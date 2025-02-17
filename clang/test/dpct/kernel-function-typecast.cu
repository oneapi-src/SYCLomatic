// RUN: dpct -out-root %T/kernel-function-typecast %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/kernel-function-typecast/kernel-function-typecast.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/kernel-function-typecast/kernel-function-typecast.dp.cpp -o %T/kernel-function-typecast/kernel-function-typecast.dp.o %}

#include <cstdint>
#include <cuda.h>

typedef uint64_t u64;

// CHECK: void exec_kernel(dpct::kernel_function cuFunc, dpct::kernel_library cuMod, dpct::queue_ptr stream) {
void exec_kernel(CUfunction cuFunc, CUmodule cuMod, CUstream stream) {
  u64 mod;
  u64 function;

  // verify the conversion from dpct::kernel_library to uint64_t
  mod = (u64)cuMod;

  // verify the conversion from uint64_t to dpct::kernel_library
  // CHECK: cuFunc = dpct::get_kernel_function((dpct::kernel_library)mod, "kfoo");
  cuModuleGetFunction(&cuFunc, (CUmodule)mod, "kfoo");

  // verify the conversion from dpct::kernel_function to uint64_t
  function = (u64)cuFunc;

  void *config[] = {0};

  // verify the conversion from uint64_t to dpct::kernel_function
  // CHECK: dpct::invoke_kernel_function((dpct::kernel_function)function, *stream, sycl::range<3>(100, 100, 100), sycl::range<3>(100, 100, 100), 1024, NULL, config);
  cuLaunchKernel((CUfunction)function, 100, 100, 100, 100, 100, 100, 1024, stream, NULL, config);
}
