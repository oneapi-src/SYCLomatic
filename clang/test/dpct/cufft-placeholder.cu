// RUN: dpct --format-range=none -out-root %T/cufft-placeholder %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-placeholder/cufft-placeholder.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>


size_t* work_size;
int odist;
int ostride;
int * onembed;
int idist;
int istride;
int* inembed;
int * n;
int rank;

//CHECK:void foo1(dpct::fft::fft_engine_ptr plan) {
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2* idata;
//CHECK-NEXT:  plan->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward);
//CHECK-NEXT:}
void foo1(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

//CHECK:void foo2(dpct::fft::fft_engine_ptr plan) {
//CHECK-NEXT:  float* odata;
//CHECK-NEXT:  sycl::float2* idata;
//CHECK-NEXT:  plan->compute<sycl::float2, float>(idata, odata, dpct::fft::fft_direction::backward);
//CHECK-NEXT:}
void foo2(cufftHandle plan) {
  float* odata;
  float2* idata;
  cufftExecC2R(plan, idata, odata);
}

int main() {
  //CHECK:dpct::fft::fft_engine_ptr plan1;
  //CHECK-NEXT:dpct::fft::fft_type type1 = dpct::fft::fft_type::complex_double_to_real_double;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan1->commit(&q_ct1, rank, n, inembed, istride, idist, onembed, ostride, odist, type1, 11, work_size);
  cufftHandle plan1;
  cufftType_t type1 = CUFFT_Z2D;
  cufftMakePlanMany(plan1, rank, n, inembed, istride, idist, onembed, ostride, odist, type1, 11, work_size);

  //CHECK:dpct::fft::fft_engine_ptr plan2;
  //CHECK-NEXT:dpct::fft::fft_type type2 = dpct::fft::fft_type::complex_float_to_real_float;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan2->commit(&q_ct1, rank, n, inembed, istride, idist, onembed, ostride, odist, type2, 12, work_size);
  cufftHandle plan2;
  cufftType_t type2 = CUFFT_C2R;
  cufftMakePlanMany(plan2, rank, n, inembed, istride, idist, onembed, ostride, odist, type2, 12, work_size);

  foo1(plan1);
  foo2(plan2);

  return 0;
}

