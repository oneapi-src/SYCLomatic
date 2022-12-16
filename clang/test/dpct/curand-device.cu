// RUN: dpct --format-range=none --usm-level=none -extra-arg-before=-std=c++14 -out-root %T/curand-device %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/curand-device/curand-device.dp.cpp --match-full-lines %s

//CHECK: #include <sycl/sycl.hpp>
//CHECK-NEXT: #include <dpct/dpct.hpp>
//CHECK-NEXT: #include <oneapi/mkl.hpp>
//CHECK-NEXT: #include <oneapi/mkl/rng/device.hpp>
//CHECK-NEXT: #include <dpct/rng_utils.hpp>
//CHECK-NEXT: #include <cstdio>
//CHECK-NEXT: #include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cstdio>

const int WARP_SIZE = 32;
const int NBLOCKS = 640;
const int ITERATIONS = 1000000;


__global__ void picount(int *totals) {
  __shared__ int counter[WARP_SIZE];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  //CHECK: dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>> rng;
  //CHECK: rng = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>(clock64(), {10, static_cast<std::uint64_t>(tid * 4)});
  curandStatePhilox4_32_10_t rng;
  curand_init(clock64(), tid, 10, &rng);

  counter[threadIdx.x] = 0;

  for (int i = 0; i < ITERATIONS; i++) {
    //CHECK: sycl::float2 x = rng.generate<oneapi::mkl::rng::device::gaussian<float>, 2>();
    //CHECK-NEXT: sycl::float2 y = rng.generate<oneapi::mkl::rng::device::gaussian<float>, 2>();
    float2 x = curand_normal2(&rng);
    float2 y = curand_normal2(&rng);
    counter[threadIdx.x] += 1 - int(x.x * x.x + y.y * y.y);
  }

  if (threadIdx.x == 0) {
    totals[blockIdx.x] = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
      totals[blockIdx.x] += counter[i];
    }
  }
}

//CHECK: void cuda_kernel_initRND(unsigned long seed, dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>> *States,
//CHECK-NEXT:                     sycl::nd_item<3> item_ct1)
__global__ void cuda_kernel_initRND(unsigned long seed, curandStateMRG32k3a_t *States)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int id    = bid*32 + tid;
  int pixel = bid*32 + tid;

  //CHECK: States[id] = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>>(seed, {10, static_cast<std::uint64_t>(pixel * 8)});
  curand_init(seed, pixel, 10, &States[id]);
}

//CHECK: void cuda_kernel_RNDnormalDitribution(sycl::double2 *Image, dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>> *States,
//CHECK-NEXT:                                  sycl::nd_item<3> item_ct1)
__global__ void cuda_kernel_RNDnormalDitribution(double2 *Image, curandStateMRG32k3a_t *States)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int id    = bid*32 + tid;
  int pixel = bid*32 + tid;

  //CHECK: Image[pixel] = States[id].generate<oneapi::mkl::rng::device::gaussian<double>, 2>();
  Image[pixel] = curand_normal2_double(&States[id]);
}

int main(int argc, char **argv) {
  int *dOut;
  //CHECK:q_ct1.submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    /*
  //CHECK-NEXT:    DPCT1101:{{[0-9]+}}: 'WARP_SIZE' expression was replaced with a value. Modify the code to use original expression, provided in comments, if it is correct.
  //CHECK-NEXT:    */
  //CHECK-NEXT:    sycl::local_accessor<int, 1> counter_acc_ct1(sycl::range<1>(32/*WARP_SIZE*/), cgh);
  //CHECK-NEXT:    dpct::access_wrapper<int *> dOut_acc_ct0(dOut, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, NBLOCKS) * sycl::range<3>(1, 1, WARP_SIZE), sycl::range<3>(1, 1, WARP_SIZE)), 
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        picount(dOut_acc_ct0.get_raw_pointer(), item_ct1, counter_acc_ct1.get_pointer());
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  picount<<<NBLOCKS, WARP_SIZE>>>(dOut);

  int size = 10;
  double2 *Image;
  //CHECK: dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>> *RandomStates;
  curandStateMRG32k3a_t *RandomStates;
  void *dev;
  //CHECK: dev = dpct::dpct_malloc(size * sizeof(dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>>));
  //CHECK-NEXT: RandomStates = (dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>>*)dev;
  cudaMalloc((void**)&dev, size * sizeof(curandStateMRG32k3a_t));
  RandomStates = (curandStateMRG32k3a_t*)dev;
  
  //CHECK: q_ct1.submit(
  //CHECK-NEXT:   [&](sycl::handler &cgh) {
  //CHECK-NEXT:     dpct::access_wrapper<dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>> *> RandomStates_acc_ct1(RandomStates, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:     cgh.parallel_for(
  //CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), 
  //CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:         cuda_kernel_initRND(1234, RandomStates_acc_ct1.get_raw_pointer(), item_ct1);
  //CHECK-NEXT:       });
  //CHECK-NEXT:   });
  //CHECK-NEXT: q_ct1.submit(
  //CHECK-NEXT:   [&](sycl::handler &cgh) {
  //CHECK-NEXT:     dpct::access_wrapper<sycl::double2 *> Image_acc_ct0(Image, cgh);
  //CHECK-NEXT:     dpct::access_wrapper<dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>> *> RandomStates_acc_ct1(RandomStates, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:     cgh.parallel_for(
  //CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), 
  //CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:         cuda_kernel_RNDnormalDitribution(Image_acc_ct0.get_raw_pointer(), RandomStates_acc_ct1.get_raw_pointer(), item_ct1);
  //CHECK-NEXT:       });
  //CHECK-NEXT:   });
  cuda_kernel_initRND<<<16,32>>>(1234, RandomStates);
  cuda_kernel_RNDnormalDitribution<<<16,32>>>(Image, RandomStates);

  return 0;
}

// CHECK: void my_kernel5(          int &a  ) {
__global__ void my_kernel5(          void  ) {
  __shared__ int a;
}

int foo() {
  int size = 10;
  //CHECK: dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>> *RandomStates;
  curandStateMRG32k3a_t *RandomStates;
  //CHECK: RandomStates = (dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>> *)dpct::dpct_malloc(size * sizeof(dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>>));
  cudaMalloc((void**)&RandomStates, size * sizeof(curandStateMRG32k3a_t));

  //CHECK: dpct::get_default_queue().submit(
  //CHECK-NEXT:   [&](sycl::handler &cgh) {
  //CHECK-NEXT:     auto RandomStates_acc_ct1 = dpct::get_access(RandomStates, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:     cgh.parallel_for(
  //CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  //CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:         cuda_kernel_initRND(1234, (dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>> *)(&RandomStates_acc_ct1[0]), item_ct1);
  //CHECK-NEXT:       });
  //CHECK-NEXT:   });
  cuda_kernel_initRND<<<16,32>>>(1234, RandomStates);

  return 0;
}

//CHECK:void kernel(dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>> *state) {
//CHECK-NEXT:  *state = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>(1111, {0, 2222 * 8});
//CHECK-NEXT:  float rand = state->generate<oneapi::mkl::rng::device::uniform<float>, 1>();
//CHECK-NEXT:}
__global__ void kernel(curandState *state) {
  curand_init(1111, 2222, 0, state);
  float rand = curand_uniform(state);
}

// Test description:
// Skip the matched TypeLoc in the implicit assignment method of RNGState.
// If tool does not skip it, the class name "state_struct_t" will be replaced with dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>>.
//     CHECK:using state_type = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>>;
//CHECK-NEXT:struct state_struct_t {
//CHECK-NEXT:  state_type state;
//CHECK-NEXT:};
//CHECK-NEXT:struct TEST {
//CHECK-NEXT:  state_struct_t rng;
//CHECK-NEXT:  TEST() {
//CHECK-NEXT:    state_struct_t rng1;
//CHECK-NEXT:    rng1 = rng;
//CHECK-NEXT:  }
//CHECK-NEXT:};
using state_type = curandStateMRG32k3a;
struct state_struct_t {
  state_type state;
};
struct TEST {
  state_struct_t rng;
  TEST() {
    state_struct_t rng1;
    rng1 = rng;
  }
};

// Test description:
// This test covers the case when the type of the arg has alias.
__device__ void foo() {
  unsigned long long seed;
  unsigned long long sequence;
  unsigned long long offset;
  state_struct_t state;
  // CHECK: state.state = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<4>>(seed, {static_cast<std::uint64_t>(offset), static_cast<std::uint64_t>(sequence * 8)});
  curand_init(seed, sequence, offset, &state.state);
}
