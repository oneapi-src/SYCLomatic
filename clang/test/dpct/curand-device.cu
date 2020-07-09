// RUN: dpct --format-range=none --usm-level=none -extra-arg-before=-std=c++14 -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/curand-device.dp.cpp --match-full-lines %s

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

  //CHECK: oneapi::mkl::rng::device::philox4x32x10<2> rng;
  //CHECK: rng = oneapi::mkl::rng::device::philox4x32x10<2>(clock64(), {10 * 2, static_cast<std::uint64_t>(tid * 8)});
  curandStatePhilox4_32_10_t rng;
  curand_init(clock64(), tid, 10, &rng);

  counter[threadIdx.x] = 0;

  for (int i = 0; i < ITERATIONS; i++) {
    //CHECK: oneapi::mkl::rng::device::uniform<float> distr_ct{{[0-9]+}};
    //CHECK-NEXT: sycl::float2 x = oneapi::mkl::rng::device::generate(distr_ct{{[0-9]+}}, rng);
    //CHECK-NEXT: sycl::float2 y = oneapi::mkl::rng::device::generate(distr_ct{{[0-9]+}}, rng);
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

//CHECK: void cuda_kernel_initRND(unsigned long seed, oneapi::mkl::rng::device::mrg32k3a<2> *States,
//CHECK-NEXT:                     sycl::nd_item<3> item_ct1)
__global__ void cuda_kernel_initRND(unsigned long seed, curandStateMRG32k3a_t *States)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int id    = bid*32 + tid;
  int pixel = bid*32 + tid;

  //CHECK: States[id] = oneapi::mkl::rng::device::mrg32k3a<2>(seed, {10 * 2, static_cast<std::uint64_t>(pixel * 8)});
  curand_init(seed, pixel, 10, &States[id]);
}

//CHECK: void cuda_kernel_RNDnormalDitribution(sycl::double2 *Image, oneapi::mkl::rng::device::mrg32k3a<2> *States,
//CHECK-NEXT:                                  sycl::nd_item<3> item_ct1)
__global__ void cuda_kernel_RNDnormalDitribution(double2 *Image, curandStateMRG32k3a_t *States)
{
  //CHECK: oneapi::mkl::rng::device::uniform<double> distr_ct{{[0-9]+}};
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int id    = bid*32 + tid;
  int pixel = bid*32 + tid;

  //CHECK: Image[pixel] = oneapi::mkl::rng::device::generate(distr_ct{{[0-9]+}}, States[id]);
  Image[pixel] = curand_normal2_double(&States[id]);
}

int main(int argc, char **argv) {
  int *dOut;
  picount<<<NBLOCKS, WARP_SIZE>>>(dOut);

  int size = 10;
  double2 *Image;
  //CHECK: oneapi::mkl::rng::device::mrg32k3a<2> *RandomStates;
  curandStateMRG32k3a_t *RandomStates;
  void *dev;
  //CHECK: dpct::dpct_malloc((void**)&dev, size * sizeof(oneapi::mkl::rng::device::mrg32k3a<2>));
  //CHECK-NEXT: RandomStates = (oneapi::mkl::rng::device::mrg32k3a<2>*)dev;
  cudaMalloc((void**)&dev, size * sizeof(curandStateMRG32k3a_t));
  RandomStates = (curandStateMRG32k3a_t*)dev;
  
  //CHECK: {
  //CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> RandomStates_buf_ct1 = dpct::get_buffer_and_offset(RandomStates);
  //CHECK-NEXT:   size_t RandomStates_offset_ct1 = RandomStates_buf_ct1.second;
  //CHECK-NEXT:   q_ct1.submit(
  //CHECK-NEXT:     [&](sycl::handler &cgh) {
  //CHECK-NEXT:       auto RandomStates_acc_ct1 = RandomStates_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:       cgh.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), 
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           oneapi::mkl::rng::device::mrg32k3a<2> *RandomStates_ct1 = (oneapi::mkl::rng::device::mrg32k3a<2> *)(&RandomStates_acc_ct1[0] + RandomStates_offset_ct1);
  //CHECK-NEXT:           cuda_kernel_initRND(1234, RandomStates_ct1, item_ct1);
  //CHECK-NEXT:         });
  //CHECK-NEXT:     });
  //CHECK-NEXT: }
  //CHECK-NEXT: {
  //CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> Image_buf_ct0 = dpct::get_buffer_and_offset(Image);
  //CHECK-NEXT:   size_t Image_offset_ct0 = Image_buf_ct0.second;
  //CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> RandomStates_buf_ct1 = dpct::get_buffer_and_offset(RandomStates);
  //CHECK-NEXT:   size_t RandomStates_offset_ct1 = RandomStates_buf_ct1.second;
  //CHECK-NEXT:   q_ct1.submit(
  //CHECK-NEXT:     [&](sycl::handler &cgh) {
  //CHECK-NEXT:       auto Image_acc_ct0 = Image_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  //CHECK-NEXT:       auto RandomStates_acc_ct1 = RandomStates_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:       cgh.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)), 
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           sycl::double2 *Image_ct0 = (sycl::double2 *)(&Image_acc_ct0[0] + Image_offset_ct0);
  //CHECK-NEXT:           oneapi::mkl::rng::device::mrg32k3a<2> *RandomStates_ct1 = (oneapi::mkl::rng::device::mrg32k3a<2> *)(&RandomStates_acc_ct1[0] + RandomStates_offset_ct1);
  //CHECK-NEXT:           cuda_kernel_RNDnormalDitribution(Image_ct0, RandomStates_ct1, item_ct1);
  //CHECK-NEXT:         });
  //CHECK-NEXT:     });
  //CHECK-NEXT: }
  cuda_kernel_initRND<<<16,32>>>(1234, RandomStates);
  cuda_kernel_RNDnormalDitribution<<<16,32>>>(Image, RandomStates);

  return 0;
}