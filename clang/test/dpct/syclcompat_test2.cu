// RUN: dpct --use-syclcompat --format-range=none --out-root %T/syclcompat_test2 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/syclcompat_test2/syclcompat_test2.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -DBUILD_TEST -c -fsycl %T/syclcompat_test2/syclcompat_test2.dp.cpp -o %T/syclcompat_test2/syclcompat_test2.dp.o %}

#include <cufft.h>
#include <curand.h>

void f1_1() {
  cufftHandle plan1;
  size_t* work_size;
  int odist;
  int ostride;
  int * onembed;
  int idist;
  int istride;
  int* inembed;
  int * n;
  // CHECK: plan1->commit(&syclcompat::get_current_device().default_queue(), 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size);
  cufftMakePlanMany(plan1, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

void f1_2() {
  float *f;
  // CHECK: f = sycl::malloc_device<float>(1, q_ct1);
  cudaMalloc(&f, sizeof(float));
  cufftHandle plan1;
  size_t* work_size;
  int odist;
  int ostride;
  int * onembed;
  int idist;
  int istride;
  int* inembed;
  int * n;
  // CHECK: plan1->commit(&q_ct1, 3, n, inembed, istride, idist, onembed, ostride, odist, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size);
  cufftMakePlanMany(plan1, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

void f2() {
  curandRngType_t rngT;
  curandGenerator_t rng;
  // CHECK: rng = dpct::rng::create_host_rng(rngT, *syclcompat::cpu_device().default_queue());
  curandCreateGeneratorHost(&rng, rngT);
}

__constant__ float const_float[10][10];

// CHECK: void k3(syclcompat::accessor<float, syclcompat::memory_region::constant, 2> const_float) {
__global__ void k3() {
  float ff = const_float[1][1];
  double d;
  sincospi(d, &d, &d);
}

void f3() {
  k3<<<1, 1>>>();
}

void f4() {
  int atomicSupported;
  int dev_id = 0;
  // CHECK: atomicSupported = syclcompat::get_device(dev_id).is_native_host_atomic_supported();
  cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, dev_id);
}

void f5() {
  float *f;
// CHECK: #if DPCT_COMPAT_RT_VERSION
// CHECK-NEXT:   f = (float *)sycl::malloc_device(4, syclcompat::get_default_queue());
// CHECK-NEXT: #endif
#if CUDART_VERSION
  cudaMalloc(&f, 4);
#endif
}
