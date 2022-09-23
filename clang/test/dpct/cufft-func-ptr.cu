// RUN: cat %s > %T/cufft-func-ptr.cu
// RUN: cd %T
// RUN: dpct -out-root %T/cufft-func-ptr cufft-func-ptr.cu --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-func-ptr/cufft-func-ptr.dp.cpp --match-full-lines cufft-func-ptr.cu
#include <cufft.h>

//CHECK:static int (*pt2CufftExec)(dpct::fft::fft_engine *, sycl::double2 *, double *) =
//CHECK-NEXT:    [](dpct::fft::fft_engine *engine, sycl::double2 *in, double *out) {
//CHECK-NEXT:      engine->compute<sycl::double2, double>(
//CHECK-NEXT:        in, out, dpct::fft::fft_direction::backward);
//CHECK-NEXT:      return 0;
//CHECK-NEXT:    };
static cufftResult (*pt2CufftExec)(cufftHandle, cufftDoubleComplex *,
                                    double *) = &cufftExecZ2D;

int main() {
//CHECK:  dpct::fft::fft_engine *plan1;
//CHECK-NEXT:  plan1 = dpct::fft::fft_engine::create(
//CHECK-NEXT:      &dpct::get_default_queue(), 10,
//CHECK-NEXT:      dpct::fft::fft_type::complex_double_to_real_double, 1);
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2 *idata;
//CHECK-NEXT:  pt2CufftExec(plan1, idata, odata);
//CHECK-NEXT:  return 0;
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);
  double* odata;
  double2* idata;
  pt2CufftExec(plan1, idata, odata);
  return 0;
}

int foo1() {
//CHECK:  typedef int (*Func_t)(dpct::fft::fft_engine *, sycl::double2 *, double *);
  typedef cufftResult (*Func_t)(cufftHandle, cufftDoubleComplex *, double *);

//     CHECK:  static Func_t FuncPtr = [](dpct::fft::fft_engine *engine, sycl::double2 *in,
//CHECK-NEXT:                             double *out) {
//CHECK-NEXT:    engine->compute<sycl::double2, double>(in, out,
//CHECK-NEXT:                                           dpct::fft::fft_direction::backward);
//CHECK-NEXT:    return 0;
//CHECK-NEXT:  };
  static Func_t FuncPtr  = &cufftExecZ2D;

//CHECK:  dpct::fft::fft_engine *plan1;
//CHECK-NEXT:  plan1 = dpct::fft::fft_engine::create(
//CHECK-NEXT:      &dpct::get_default_queue(), 10,
//CHECK-NEXT:      dpct::fft::fft_type::complex_double_to_real_double, 1);
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2 *idata;
//CHECK-NEXT:  FuncPtr(plan1, idata, odata);
//CHECK-NEXT:  return 0;
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);
  double* odata;
  double2* idata;
  FuncPtr(plan1, idata, odata);
  return 0;
}

int foo2() {
//CHECK:  using Func_t = int (*)(dpct::fft::fft_engine *, sycl::double2 *, double *);
  using Func_t = cufftResult (*)(cufftHandle, cufftDoubleComplex *, double *);

//     CHECK:  Func_t FuncPtr2 = [](dpct::fft::fft_engine *engine, sycl::double2 *in,
//CHECK-NEXT:                       double *out) {
//CHECK-NEXT:    engine->compute<sycl::double2, double>(in, out,
//CHECK-NEXT:                                           dpct::fft::fft_direction::backward);
//CHECK-NEXT:    return 0;
//CHECK-NEXT:  };
  Func_t FuncPtr2  = &cufftExecZ2D;

//CHECK:  dpct::fft::fft_engine *plan1;
//CHECK-NEXT:  plan1 = dpct::fft::fft_engine::create(
//CHECK-NEXT:      &dpct::get_default_queue(), 10,
//CHECK-NEXT:      dpct::fft::fft_type::complex_double_to_real_double, 1);
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2 *idata;
//CHECK-NEXT:  FuncPtr2(plan1, idata, odata);
//CHECK-NEXT:  return 0;
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);
  double* odata;
  double2* idata;
  FuncPtr2(plan1, idata, odata);
  return 0;
}

int foo3() {
//CHECK:  using Func_t = int (*)(dpct::fft::fft_engine *, sycl::double2 *, double *);
  using Func_t = cufftResult (*)(cufftHandle, cufftDoubleComplex *, double *);

//CHECK:  Func_t FuncPtr3;
//CHECK-NEXT:  FuncPtr3 = [](dpct::fft::fft_engine *engine, sycl::double2 *in, double *out) {
//CHECK-NEXT:    engine->compute<sycl::double2, double>(in, out,
//CHECK-NEXT:                                           dpct::fft::fft_direction::backward);
//CHECK-NEXT:    return 0;
//CHECK-NEXT:  };
  Func_t FuncPtr3;
  FuncPtr3 = &cufftExecZ2D;

//CHECK:  dpct::fft::fft_engine *plan1;
//CHECK-NEXT:  plan1 = dpct::fft::fft_engine::create(
//CHECK-NEXT:      &dpct::get_default_queue(), 10, 
//CHECK-NEXT:      dpct::fft::fft_type::complex_double_to_real_double, 1);
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2 *idata;
//CHECK-NEXT:  FuncPtr3(plan1, idata, odata);
//CHECK-NEXT:  return 0;
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);
  double* odata;
  double2* idata;
  FuncPtr3(plan1, idata, odata);
  return 0;
}

int foo4() {
//CHECK:  int (*FuncPtr4)(dpct::fft::fft_engine *, sycl::double2 *, double *);
  cufftResult (*FuncPtr4)(cufftHandle, cufftDoubleComplex *, double *);

//CHECK:  FuncPtr4 = [](dpct::fft::fft_engine *engine, sycl::double2 *in, double *out) {
//CHECK-NEXT:    engine->compute<sycl::double2, double>(in, out,
//CHECK-NEXT:                                           dpct::fft::fft_direction::backward);
//CHECK-NEXT:    return 0;
//CHECK-NEXT:  };
  FuncPtr4 = &cufftExecZ2D;

//CHECK:  dpct::fft::fft_engine *plan1;
//CHECK-NEXT:  plan1 = dpct::fft::fft_engine::create(
//CHECK-NEXT:      &dpct::get_default_queue(), 10,
//CHECK-NEXT:      dpct::fft::fft_type::complex_double_to_real_double, 1);
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2 *idata;
//CHECK-NEXT:  FuncPtr4(plan1, idata, odata);
//CHECK-NEXT:  return 0;
  cufftHandle plan1;
  cufftPlan1d(&plan1, 10, CUFFT_Z2D, 1);
  double* odata;
  double2* idata;
  FuncPtr4(plan1, idata, odata);
  return 0;
}