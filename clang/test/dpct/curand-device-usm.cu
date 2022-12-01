// RUN: dpct --format-range=none -extra-arg-before=-std=c++14 -out-root %T/curand-device-usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/curand-device-usm/curand-device-usm.dp.cpp --match-full-lines %s

//CHECK: #include <sycl/sycl.hpp>
//CHECK-NEXT: #include <dpct/dpct.hpp>
//CHECK-NEXT: #include <oneapi/mkl.hpp>
//CHECK-NEXT: #include <oneapi/mkl/rng/device.hpp>
//CHECK-NEXT: #include <dpct/rng_utils.hpp>
//CHECK-NEXT: #include <cstdio>
//CHECK-NEXT: #include <tuple>
//CHECK-NEXT: #include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cstdio>
#include <tuple>

const int WARP_SIZE = 32;
const int NBLOCKS = 640;
const int ITERATIONS = 1000000;


__global__ void picount(int *totals) {
  __shared__ int counter[WARP_SIZE];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  //CHECK: dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> rng;
  //CHECK: rng = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>(clock64(), {0, static_cast<std::uint64_t>(tid * 8)});
  curandState_t rng;
  curand_init(clock64(), tid, 0, &rng);

  counter[threadIdx.x] = 0;

  for (int i = 0; i < ITERATIONS; i++) {
    //CHECK: float x = rng.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
    //CHECK-NEXT: float y = rng.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
    float x = curand_uniform(&rng);
    float y = curand_uniform(&rng);
    counter[threadIdx.x] += 1 - int(x * x + y * y);
  }

  if (threadIdx.x == 0) {
    totals[blockIdx.x] = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
      totals[blockIdx.x] += counter[i];
    }
  }
}

//CHECK: void cuda_kernel_initRND(unsigned long seed, dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *States,
//CHECK-NEXT:                     sycl::nd_item<3> item_ct1)
__global__ void cuda_kernel_initRND(unsigned long seed, curandState *States)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int id    = bid*32 + tid;
  int pixel = bid*32 + tid;

  //CHECK: States[id] = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>(seed, {0, static_cast<std::uint64_t>(pixel * 8)});
  curand_init(seed, pixel, 0, &States[id]);
}

//CHECK: void cuda_kernel_RNDnormalDitribution(double *Image, dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *States,
//CHECK-NEXT:                                  sycl::nd_item<3> item_ct1)
__global__ void cuda_kernel_RNDnormalDitribution(double *Image, curandState *States)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int id    = bid*32 + tid;
  int pixel = bid*32 + tid;

  //CHECK: Image[pixel] = States[id].generate<oneapi::mkl::rng::device::gaussian<double>, 1>();
  Image[pixel] = curand_normal_double(&States[id]);
}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

int main(int argc, char **argv) {
  int *dOut;
  picount<<<NBLOCKS, WARP_SIZE>>>(dOut);

  int size = 10;
  double *Image;
  //CHECK: dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *RandomStates;
  curandState *RandomStates;
  void *dev;
  //CHECK: dev = (void *)sycl::malloc_device(size * sizeof(dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>), q_ct1);
  cudaMalloc((void**)&dev, size * sizeof(curandState));
  //CHECK: RandomStates = (dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>*)dev;
  RandomStates = (curandState*)dev;
  //CHECK: RandomStates = (dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *)sycl::malloc_device(size * sizeof(dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>) * 10, q_ct1);
  cudaMalloc((void**)&RandomStates, size * sizeof(curandState) * 10);
  //CHECK: RandomStates = sycl::malloc_device<dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>>(size, q_ct1);
  cudaMalloc((void**)&RandomStates, size * sizeof(curandState));

  cuda_kernel_initRND<<<16,32>>>(1234, RandomStates);
  cuda_kernel_RNDnormalDitribution<<<16,32>>>(Image, RandomStates);

  //CHECK: CHECK((dOut = sycl::malloc_device<int>(10, q_ct1), 0));
  CHECK(cudaMalloc((void **)&dOut, sizeof(int) * 10));
  //CHECK: CHECK((RandomStates = (dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> *)sycl::malloc_device(sizeof(dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>) * 10 * 10, q_ct1), 0));
  CHECK(cudaMalloc((void **)&RandomStates, sizeof(curandState) * 10 * 10));
  //CHECK: sycl::range<3> grid(1, 1, 10);
  dim3 grid(10, 1);
  //CHECK: CHECK((dOut = sycl::malloc_device<int>(grid[2], q_ct1), 0));
  CHECK(cudaMalloc((void **)&dOut, sizeof(int) * grid.x));

  return 0;
}


__global__ void test() {
  std::tuple<unsigned int, unsigned int> seeds = {1, 2};
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  //CHECK:state = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(std::get<0>(seeds), {static_cast<std::uint64_t>(std::get<1>(seeds)), static_cast<std::uint64_t>(idx * 4)});
  curand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);
}

// Test description:
// This test covers the case when the type of the arg has alias.
// CHECK: using state_type = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>>;
using state_type = curandStateMRG32k3a;
struct state_struct_t {
  state_type state;
};
__device__ void foo() {
  unsigned long long seed;
  unsigned long long sequence;
  unsigned long long offset;
  state_struct_t state;
  // CHECK: state.state = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>>(seed, {static_cast<std::uint64_t>(offset), static_cast<std::uint64_t>(sequence * 8)});
  curand_init(seed, sequence, offset, &state.state);
}