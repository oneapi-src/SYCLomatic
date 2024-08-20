// RUN: dpct --format-range=none -out-root %T/cuda_arch_test/math_in_cuda_arch %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/cuda_arch_test/math_in_cuda_arch/math_in_cuda_arch.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda_arch_test/math_in_cuda_arch/math_in_cuda_arch.dp.cpp -o %T/cuda_arch_test/math_in_cuda_arch/math_in_cuda_arch.dp.o %}

// CHECK: float f(float x) {
// CHECK-EMPTY: 
// CHECK-NEXT:   return expf(x);
// CHECK-EMPTY: 
// CHECK-NEXT: }
// CHECK-NEXT: float f_host_ct0(float x) {
// CHECK-EMPTY: 
// CHECK-NEXT:   return sycl::exp(x);
// CHECK-EMPTY: 
// CHECK-NEXT: }
__device__ __host__ float f(float x) {
#if defined(__CUDA_ARCH__)
  return ::expf(x);
#else
  return std::exp(x);
#endif
}
