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

//CHECK:void foo1(dpct::fft::fft_engine* plan) {
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2* idata;
//CHECK-NEXT:  plan->compute<sycl::double2, double>(idata, odata, dpct::fft::fft_direction::backward);
//CHECK-NEXT:}
void foo1(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

//CHECK:void foo2(dpct::fft::fft_engine* plan) {
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

  //CHECK:dpct::fft::fft_engine* plan1;
  //CHECK-NEXT:plan1->commit(&q_ct1, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, nullptr);
  cufftHandle plan1;
  cufftMakePlanMany(plan1, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);

  //CHECK:dpct::fft::fft_engine* plan2;
  //CHECK-NEXT:plan2->commit(&q_ct1, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, nullptr);
  cufftHandle plan2;
  cufftMakePlanMany(plan2, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);

  //CHECK:dpct::fft::fft_engine* plan3;
  //CHECK-NEXT:plan3->commit(&q_ct1, rank, n, inembed, istride, idist, onembed, ostride, odist, type2, 12, nullptr);
  cufftHandle plan3;
  cufftMakePlanMany(plan3, rank, n, inembed, istride, idist, onembed, ostride, odist, type2, 12, work_size);

  foo1(plan1);
  foo2(plan2);

  return 0;
}

