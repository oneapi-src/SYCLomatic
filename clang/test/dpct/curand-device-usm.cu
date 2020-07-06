// RUN: dpct --format-range=none -extra-arg-before=-std=c++14 -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/curand-device-usm.dp.cpp --match-full-lines %s

//CHECK: #include <CL/sycl.hpp>
//CHECK-NEXT: #include <dpct/dpct.hpp>
//CHECK-NEXT: #include <mkl_rng_sycl_device.hpp>
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

  //CHECK: mkl::rng::device::philox4x32x10<1> rng;
  //CHECK: rng = mkl::rng::device::philox4x32x10<1>(clock64(), {0, static_cast<std::uint64_t>(tid * 8)});
  curandState_t rng;
  curand_init(clock64(), tid, 0, &rng);

  counter[threadIdx.x] = 0;

  for (int i = 0; i < ITERATIONS; i++) {
    //CHECK: mkl::rng::device::uniform<float> distr_ct{{[0-9]+}};
    //CHECK-NEXT: float x = mkl::rng::device::generate(distr_ct{{[0-9]+}}, rng);
    //CHECK-NEXT: float y = mkl::rng::device::generate(distr_ct{{[0-9]+}}, rng);
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

//CHECK: void cuda_kernel_initRND(unsigned long seed, mkl::rng::device::philox4x32x10<1> *States,
//CHECK-NEXT:                     sycl::nd_item<3> item_ct1)
__global__ void cuda_kernel_initRND(unsigned long seed, curandState *States)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int id    = bid*32 + tid;
  int pixel = bid*32 + tid;

  //CHECK: States[id] = mkl::rng::device::philox4x32x10<1>(seed, {0, static_cast<std::uint64_t>(pixel * 8)});
  curand_init(seed, pixel, 0, &States[id]);
}

//CHECK: void cuda_kernel_RNDnormalDitribution(double *Image, mkl::rng::device::philox4x32x10<1> *States,
//CHECK-NEXT:                                  sycl::nd_item<3> item_ct1)
__global__ void cuda_kernel_RNDnormalDitribution(double *Image, curandState *States)
{
  //CHECK: mkl::rng::device::uniform<double> distr_ct{{[0-9]+}};
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int id    = bid*32 + tid;
  int pixel = bid*32 + tid;

  //CHECK: Image[pixel] = mkl::rng::device::generate(distr_ct{{[0-9]+}}, States[id]);
  Image[pixel] = curand_normal_double(&States[id]);
}

int main(int argc, char **argv) {
  int *dOut;
  picount<<<NBLOCKS, WARP_SIZE>>>(dOut);

  int size = 10;
  double *Image;
  //CHECK: mkl::rng::device::philox4x32x10<1> *RandomStates;
  curandState *RandomStates;
  void *dev;
  //CHECK: dev = (void *)sycl::malloc_device(size * sizeof(mkl::rng::device::philox4x32x10<1>), q_ct1);
  cudaMalloc((void**)&dev, size * sizeof(curandState));
  //CHECK: RandomStates = (mkl::rng::device::philox4x32x10<1>*)dev;
  RandomStates = (curandState*)dev;
  
  cuda_kernel_initRND<<<16,32>>>(1234, RandomStates);
  cuda_kernel_RNDnormalDitribution<<<16,32>>>(Image, RandomStates);

  return 0;
}