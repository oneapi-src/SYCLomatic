// RUN: dpct --format-range=none -out-root %T/cufft-deduce %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/cufft-deduce/cufft-deduce.dp.cpp --match-full-lines %s
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
constexpr int rank = 3;

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
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2* idata;
//CHECK-NEXT:  plan->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward);
//CHECK-NEXT:}
void foo2(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

int main() {
  //CHECK:constexpr dpct::fft::fft_type type = dpct::fft::fft_type::complex_double_to_real_double;
  constexpr cufftType_t type = CUFFT_Z2D;
  cufftType_t type2 = type;

  //CHECK:dpct::fft::fft_engine_ptr plan1;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan1->commit(&q_ct1, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);
  cufftHandle plan1;
  cufftMakePlanMany(plan1, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);

  //CHECK:dpct::fft::fft_engine_ptr plan2;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan2->commit(&q_ct1, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);
  cufftHandle plan2;
  cufftMakePlanMany(plan2, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);

  //CHECK:dpct::fft::fft_engine_ptr plan3;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan3->commit(&q_ct1, rank, n, inembed, istride, idist, onembed, ostride, odist, type2, 12, work_size);
  cufftHandle plan3;
  cufftMakePlanMany(plan3, rank, n, inembed, istride, idist, onembed, ostride, odist, type2, 12, work_size);

  foo1(plan1);
  foo2(plan2);

  return 0;
}

