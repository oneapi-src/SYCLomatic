// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/kernel_lambda_arg %s --usm-level=restricted --cuda-include-path="%cuda-path/include" --sycl-named-lambda
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel_lambda_arg/kernel_lambda_arg.dp.cpp

#include <curand_kernel.h>
#include <tuple>

// The kernel function can be migrated on windows with v8.0 SDK with -fno-delayed-template-parsing option.
// But the kernel call which has device lambda as argument has a parsing error on windows with v8.0 SDK.
// So we disable this test on windows with v8.0 SDK.

//CHECK:template <typename T>
//CHECK-NEXT:void my_kernel1(const T func) {
//CHECK-NEXT:  func(10);
//CHECK-NEXT:}
template <typename T>
__global__ void my_kernel1(const T func) {
  func(10);
}

//CHECK:void run_foo1() {
//CHECK-NEXT:  dpct::get_default_queue().parallel_for<dpct_kernel_name<class my_kernel1_{{[0-9a-z]+}}, class lambda_{{[0-9a-z]+}}>>(
//CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:      my_kernel1([=] (int idx) { idx++; });
//CHECK-NEXT:    });
//CHECK-NEXT:}
void run_foo1() {
  my_kernel1<<<1, 1>>>([=] __device__(int idx) { idx++; });
}

//     CHECK:template <typename Foo> void my_kernel2(const Foo &foo,
//CHECK-NEXT:                                        sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  std::tuple<unsigned int, unsigned int> seeds = {1, 2};
//CHECK-NEXT:  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
//CHECK-NEXT:  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>> state;
//CHECK-NEXT:  state = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>(std::get<0>(seeds), {static_cast<std::uint64_t>(std::get<1>(seeds)), static_cast<std::uint64_t>(idx * 4)});
//CHECK-NEXT:  foo(&state);
//CHECK-NEXT:}
template <typename Foo> __global__ void my_kernel2(const Foo &foo) {
  std::tuple<unsigned int, unsigned int> seeds = {1, 2};
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);
  foo(&state);
}

//     CHECK:inline void run_foo2() {
//CHECK-NEXT:  dpct::get_default_queue().parallel_for<dpct_kernel_name<class my_kernel2_{{[0-9a-z]+}}, class lambda_{{[0-9a-z]+}}>>(
//CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
//CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:      my_kernel2([] (dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>> * state) {
//CHECK-NEXT:    return state->generate<oneapi::mkl::rng::device::uniform<double>, 2>();
//CHECK-NEXT:  }, item_ct1);
//CHECK-NEXT:    });
//CHECK-NEXT:}
inline void run_foo2() {
  my_kernel2<<<1, 1>>>([] __device__(curandStatePhilox4_32_10_t * state) {
    return curand_uniform2_double(state);
  });
}
