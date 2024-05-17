// UNSUPPORTED: system-windows
// RUN: dpct --format-range=none -out-root %T/cuda_arch_test %s --cuda-include-path="%cuda-path/include" --stop-on-parse-err -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda_arch_test/cuda_arch_test.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda_arch_test/cuda_arch_test.dp.cpp -o %T/cuda_arch_test/cuda_arch_test.dp.o %}
#include <cuda_runtime.h>
// CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
#ifdef __CUDA_ARCH__
// CHECK: #define GMX_PTX_ARCH DPCT_COMPATIBILITY_TEMP
#define GMX_PTX_ARCH __CUDA_ARCH__
#else 
#define GMX_PTX_ARCH 0
#endif
__global__ void test() {
constexpr bool c_preloadCj = GMX_PTX_ARCH < 700;
}
