// RUN: dpct --format-range=none -out-root %T/cufft-usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-usm/cufft-usm.dp.cpp --match-full-lines %s
// CHECK: #include <dpct/lib_common_utils.hpp>
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>


int main() {
  //CHECK:dpct::fft::fft_engine_ptr plan_1d_C2C;
  //CHECK-NEXT:sycl::float2* odata_1d_C2C;
  //CHECK-NEXT:sycl::float2* idata_1d_C2C;
  //CHECK-NEXT:plan_1d_C2C = dpct::fft::fft_engine::create(&q_ct1, 10, dpct::fft::fft_type::complex_float_to_complex_float, 3);
  //CHECK-NEXT:plan_1d_C2C->compute<sycl::float2, sycl::float2>(idata_1d_C2C, odata_1d_C2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_1d_C2C;
  float2* odata_1d_C2C;
  float2* idata_1d_C2C;
  cufftPlan1d(&plan_1d_C2C, 10, CUFFT_C2C, 3);
  cufftExecC2C(plan_1d_C2C, idata_1d_C2C, odata_1d_C2C, CUFFT_FORWARD);

  //CHECK:dpct::fft::fft_engine_ptr plan_1d_C2R;
  //CHECK-NEXT:float* odata_1d_C2R;
  //CHECK-NEXT:sycl::float2* idata_1d_C2R;
  //CHECK-NEXT:plan_1d_C2R = dpct::fft::fft_engine::create(&q_ct1, 10, dpct::fft::fft_type::complex_float_to_real_float, 3);
  //CHECK-NEXT:plan_1d_C2R->compute<sycl::float2, float>(idata_1d_C2R, odata_1d_C2R, dpct::fft::fft_direction::backward);
  cufftHandle plan_1d_C2R;
  float* odata_1d_C2R;
  float2* idata_1d_C2R;
  cufftPlan1d(&plan_1d_C2R, 10, CUFFT_C2R, 3);
  cufftExecC2R(plan_1d_C2R, idata_1d_C2R, odata_1d_C2R);

  //CHECK:dpct::fft::fft_engine_ptr plan_1d_R2C;
  //CHECK-NEXT:sycl::float2* odata_1d_R2C;
  //CHECK-NEXT:float* idata_1d_R2C;
  //CHECK-NEXT:plan_1d_R2C = dpct::fft::fft_engine::create(&q_ct1, 10, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:plan_1d_R2C->compute<float, sycl::float2>(idata_1d_R2C, odata_1d_R2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_1d_R2C;
  float2* odata_1d_R2C;
  float* idata_1d_R2C;
  cufftPlan1d(&plan_1d_R2C, 10, CUFFT_R2C, 3);
  cufftExecR2C(plan_1d_R2C, idata_1d_R2C, odata_1d_R2C);

  //CHECK:dpct::fft::fft_engine_ptr plan_1d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_1d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_1d_Z2Z;
  //CHECK-NEXT:plan_1d_Z2Z = dpct::fft::fft_engine::create(&q_ct1, 10, dpct::fft::fft_type::complex_double_to_complex_double, 3);
  //CHECK-NEXT:plan_1d_Z2Z->compute<sycl::double2, sycl::double2>(idata_1d_Z2Z, odata_1d_Z2Z, dpct::fft::fft_direction::backward);
  cufftHandle plan_1d_Z2Z;
  double2* odata_1d_Z2Z;
  double2* idata_1d_Z2Z;
  cufftPlan1d(&plan_1d_Z2Z, 10, CUFFT_Z2Z, 3);
  cufftExecZ2Z(plan_1d_Z2Z, idata_1d_Z2Z, odata_1d_Z2Z, CUFFT_INVERSE);

  //CHECK:dpct::fft::fft_engine_ptr plan_1d_Z2D;
  //CHECK-NEXT:double* odata_1d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_1d_Z2D;
  //CHECK-NEXT:plan_1d_Z2D = dpct::fft::fft_engine::create(&q_ct1, 10, dpct::fft::fft_type::complex_double_to_real_double, 3);
  //CHECK-NEXT:plan_1d_Z2D->compute<sycl::double2, double>(idata_1d_Z2D, odata_1d_Z2D, dpct::fft::fft_direction::backward);
  cufftHandle plan_1d_Z2D;
  double* odata_1d_Z2D;
  double2* idata_1d_Z2D;
  cufftPlan1d(&plan_1d_Z2D, 10, CUFFT_Z2D, 3);
  cufftExecZ2D(plan_1d_Z2D, idata_1d_Z2D, odata_1d_Z2D);

  //CHECK:dpct::fft::fft_engine_ptr plan_1d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_1d_D2Z;
  //CHECK-NEXT:double* idata_1d_D2Z;
  //CHECK-NEXT:plan_1d_D2Z = dpct::fft::fft_engine::create(&q_ct1, 10, dpct::fft::fft_type::real_double_to_complex_double, 3);
  //CHECK-NEXT:plan_1d_D2Z->compute<double, sycl::double2>(idata_1d_D2Z, odata_1d_D2Z, dpct::fft::fft_direction::forward);
  cufftHandle plan_1d_D2Z;
  double2* odata_1d_D2Z;
  double* idata_1d_D2Z;
  cufftPlan1d(&plan_1d_D2Z, 10, CUFFT_D2Z, 3);
  cufftExecD2Z(plan_1d_D2Z, idata_1d_D2Z, odata_1d_D2Z);

  //CHECK:dpct::fft::fft_engine_ptr plan_2d_C2C;
  //CHECK-NEXT:sycl::float2* odata_2d_C2C;
  //CHECK-NEXT:sycl::float2* idata_2d_C2C;
  //CHECK-NEXT:plan_2d_C2C = dpct::fft::fft_engine::create(&q_ct1, 10, 20, dpct::fft::fft_type::complex_float_to_complex_float);
  //CHECK-NEXT:plan_2d_C2C->compute<sycl::float2, sycl::float2>(idata_2d_C2C, odata_2d_C2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_2d_C2C;
  float2* odata_2d_C2C;
  float2* idata_2d_C2C;
  cufftPlan2d(&plan_2d_C2C, 10, 20, CUFFT_C2C);
  cufftExecC2C(plan_2d_C2C, idata_2d_C2C, odata_2d_C2C, CUFFT_FORWARD);

  //CHECK:dpct::fft::fft_engine_ptr plan_2d_C2R;
  //CHECK-NEXT:float* odata_2d_C2R;
  //CHECK-NEXT:sycl::float2* idata_2d_C2R;
  //CHECK-NEXT:plan_2d_C2R = dpct::fft::fft_engine::create(&q_ct1, 10, 20, dpct::fft::fft_type::complex_float_to_real_float);
  //CHECK-NEXT:plan_2d_C2R->compute<sycl::float2, float>(idata_2d_C2R, odata_2d_C2R, dpct::fft::fft_direction::backward);
  cufftHandle plan_2d_C2R;
  float* odata_2d_C2R;
  float2* idata_2d_C2R;
  cufftPlan2d(&plan_2d_C2R, 10, 20, CUFFT_C2R);
  cufftExecC2R(plan_2d_C2R, idata_2d_C2R, odata_2d_C2R);

  //CHECK:dpct::fft::fft_engine_ptr plan_2d_R2C;
  //CHECK-NEXT:sycl::float2* odata_2d_R2C;
  //CHECK-NEXT:float* idata_2d_R2C;
  //CHECK-NEXT:plan_2d_R2C = dpct::fft::fft_engine::create(&q_ct1, 10, 20, dpct::fft::fft_type::real_float_to_complex_float);
  //CHECK-NEXT:plan_2d_R2C->compute<float, sycl::float2>(idata_2d_R2C, odata_2d_R2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_2d_R2C;
  float2* odata_2d_R2C;
  float* idata_2d_R2C;
  cufftPlan2d(&plan_2d_R2C, 10, 20, CUFFT_R2C);
  cufftExecR2C(plan_2d_R2C, idata_2d_R2C, odata_2d_R2C);

  //CHECK:dpct::fft::fft_engine_ptr plan_2d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_2d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_2d_Z2Z;
  //CHECK-NEXT:plan_2d_Z2Z = dpct::fft::fft_engine::create(&q_ct1, 10, 20, dpct::fft::fft_type::complex_double_to_complex_double);
  //CHECK-NEXT:plan_2d_Z2Z->compute<sycl::double2, sycl::double2>(idata_2d_Z2Z, odata_2d_Z2Z, dpct::fft::fft_direction::backward);
  cufftHandle plan_2d_Z2Z;
  double2* odata_2d_Z2Z;
  double2* idata_2d_Z2Z;
  cufftPlan2d(&plan_2d_Z2Z, 10, 20, CUFFT_Z2Z);
  cufftExecZ2Z(plan_2d_Z2Z, idata_2d_Z2Z, odata_2d_Z2Z, CUFFT_INVERSE);

  //CHECK:dpct::fft::fft_engine_ptr plan_2d_Z2D;
  //CHECK-NEXT:double* odata_2d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_2d_Z2D;
  //CHECK-NEXT:plan_2d_Z2D = dpct::fft::fft_engine::create(&q_ct1, 10, 20, dpct::fft::fft_type::complex_double_to_real_double);
  //CHECK-NEXT:plan_2d_Z2D->compute<sycl::double2, double>(idata_2d_Z2D, odata_2d_Z2D, dpct::fft::fft_direction::backward);
  cufftHandle plan_2d_Z2D;
  double* odata_2d_Z2D;
  double2* idata_2d_Z2D;
  cufftPlan2d(&plan_2d_Z2D, 10, 20, CUFFT_Z2D);
  cufftExecZ2D(plan_2d_Z2D, idata_2d_Z2D, odata_2d_Z2D);

  //CHECK:dpct::fft::fft_engine_ptr plan_2d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_2d_D2Z;
  //CHECK-NEXT:double* idata_2d_D2Z;
  //CHECK-NEXT:plan_2d_D2Z = dpct::fft::fft_engine::create(&q_ct1, 10, 20, dpct::fft::fft_type::real_double_to_complex_double);
  //CHECK-NEXT:plan_2d_D2Z->compute<double, sycl::double2>(idata_2d_D2Z, odata_2d_D2Z, dpct::fft::fft_direction::forward);
  cufftHandle plan_2d_D2Z;
  double2* odata_2d_D2Z;
  double* idata_2d_D2Z;
  cufftPlan2d(&plan_2d_D2Z, 10, 20, CUFFT_D2Z);
  cufftExecD2Z(plan_2d_D2Z, idata_2d_D2Z, odata_2d_D2Z);

  //CHECK:dpct::fft::fft_engine_ptr plan_3d_C2C;
  //CHECK-NEXT:sycl::float2* odata_3d_C2C;
  //CHECK-NEXT:sycl::float2* idata_3d_C2C;
  //CHECK-NEXT:plan_3d_C2C = dpct::fft::fft_engine::create(&q_ct1, 10, 20, 30, dpct::fft::fft_type::complex_float_to_complex_float);
  //CHECK-NEXT:plan_3d_C2C->compute<sycl::float2, sycl::float2>(idata_3d_C2C, odata_3d_C2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_3d_C2C;
  float2* odata_3d_C2C;
  float2* idata_3d_C2C;
  cufftPlan3d(&plan_3d_C2C, 10, 20, 30, CUFFT_C2C);
  cufftExecC2C(plan_3d_C2C, idata_3d_C2C, odata_3d_C2C, CUFFT_FORWARD);

  //CHECK:dpct::fft::fft_engine_ptr plan_3d_C2R;
  //CHECK-NEXT:float* odata_3d_C2R;
  //CHECK-NEXT:sycl::float2* idata_3d_C2R;
  //CHECK-NEXT:plan_3d_C2R = dpct::fft::fft_engine::create(&q_ct1, 10, 20, 30, dpct::fft::fft_type::complex_float_to_real_float);
  //CHECK-NEXT:plan_3d_C2R->compute<sycl::float2, float>(idata_3d_C2R, odata_3d_C2R, dpct::fft::fft_direction::backward);
  cufftHandle plan_3d_C2R;
  float* odata_3d_C2R;
  float2* idata_3d_C2R;
  cufftPlan3d(&plan_3d_C2R, 10, 20, 30, CUFFT_C2R);
  cufftExecC2R(plan_3d_C2R, idata_3d_C2R, odata_3d_C2R);

  //CHECK:dpct::fft::fft_engine_ptr plan_3d_R2C;
  //CHECK-NEXT:sycl::float2* odata_3d_R2C;
  //CHECK-NEXT:float* idata_3d_R2C;
  //CHECK-NEXT:plan_3d_R2C = dpct::fft::fft_engine::create(&q_ct1, 10, 20, 30, dpct::fft::fft_type::real_float_to_complex_float);
  //CHECK-NEXT:plan_3d_R2C->compute<float, sycl::float2>(idata_3d_R2C, odata_3d_R2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_3d_R2C;
  float2* odata_3d_R2C;
  float* idata_3d_R2C;
  cufftPlan3d(&plan_3d_R2C, 10, 20, 30, CUFFT_R2C);
  cufftExecR2C(plan_3d_R2C, idata_3d_R2C, odata_3d_R2C);

  //CHECK:dpct::fft::fft_engine_ptr plan_3d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_3d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_3d_Z2Z;
  //CHECK-NEXT:plan_3d_Z2Z = dpct::fft::fft_engine::create(&q_ct1, 10, 20, 30, dpct::fft::fft_type::complex_double_to_complex_double);
  //CHECK-NEXT:plan_3d_Z2Z->compute<sycl::double2, sycl::double2>(idata_3d_Z2Z, odata_3d_Z2Z, dpct::fft::fft_direction::backward);
  cufftHandle plan_3d_Z2Z;
  double2* odata_3d_Z2Z;
  double2* idata_3d_Z2Z;
  cufftPlan3d(&plan_3d_Z2Z, 10, 20, 30, CUFFT_Z2Z);
  cufftExecZ2Z(plan_3d_Z2Z, idata_3d_Z2Z, odata_3d_Z2Z, CUFFT_INVERSE);

  //CHECK:dpct::fft::fft_engine_ptr plan_3d_Z2D;
  //CHECK-NEXT:double* odata_3d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_3d_Z2D;
  //CHECK-NEXT:plan_3d_Z2D = dpct::fft::fft_engine::create(&q_ct1, 10, 20, 30, dpct::fft::fft_type::complex_double_to_real_double);
  //CHECK-NEXT:plan_3d_Z2D->compute<sycl::double2, double>(idata_3d_Z2D, odata_3d_Z2D, dpct::fft::fft_direction::backward);
  cufftHandle plan_3d_Z2D;
  double* odata_3d_Z2D;
  double2* idata_3d_Z2D;
  cufftPlan3d(&plan_3d_Z2D, 10, 20, 30, CUFFT_Z2D);
  cufftExecZ2D(plan_3d_Z2D, idata_3d_Z2D, odata_3d_Z2D);

  //CHECK:dpct::fft::fft_engine_ptr plan_3d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_3d_D2Z;
  //CHECK-NEXT:double* idata_3d_D2Z;
  //CHECK-NEXT:plan_3d_D2Z = dpct::fft::fft_engine::create(&q_ct1, 10, 20, 30, dpct::fft::fft_type::real_double_to_complex_double);
  //CHECK-NEXT:plan_3d_D2Z->compute<double, sycl::double2>(idata_3d_D2Z, odata_3d_D2Z, dpct::fft::fft_direction::forward);
  cufftHandle plan_3d_D2Z;
  double2* odata_3d_D2Z;
  double* idata_3d_D2Z;
  cufftPlan3d(&plan_3d_D2Z, 10, 20, 30, CUFFT_D2Z);
  cufftExecD2Z(plan_3d_D2Z, idata_3d_D2Z, odata_3d_D2Z);

  //CHECK:dpct::fft::fft_engine_ptr plan_many_C2C;
  //CHECK-NEXT:int odist_many_C2C;
  //CHECK-NEXT:int ostride_many_C2C;
  //CHECK-NEXT:int * onembed_many_C2C;
  //CHECK-NEXT:int idist_many_C2C;
  //CHECK-NEXT:int istride_many_C2C;
  //CHECK-NEXT:int* inembed_many_C2C;
  //CHECK-NEXT:int * n_many_C2C;
  //CHECK-NEXT:sycl::float2* odata_many_C2C;
  //CHECK-NEXT:sycl::float2* idata_many_C2C;
  //CHECK-NEXT:plan_many_C2C = dpct::fft::fft_engine::create(&q_ct1, 3, n_many_C2C, inembed_many_C2C, istride_many_C2C, idist_many_C2C, onembed_many_C2C, ostride_many_C2C, odist_many_C2C, dpct::fft::fft_type::complex_float_to_complex_float, 12);
  //CHECK-NEXT:plan_many_C2C->compute<sycl::float2, sycl::float2>(idata_many_C2C, odata_many_C2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_many_C2C;
  int odist_many_C2C;
  int ostride_many_C2C;
  int * onembed_many_C2C;
  int idist_many_C2C;
  int istride_many_C2C;
  int* inembed_many_C2C;
  int * n_many_C2C;
  float2* odata_many_C2C;
  float2* idata_many_C2C;
  cufftPlanMany(&plan_many_C2C, 3, n_many_C2C, inembed_many_C2C, istride_many_C2C, idist_many_C2C, onembed_many_C2C, ostride_many_C2C, odist_many_C2C, CUFFT_C2C, 12);
  cufftExecC2C(plan_many_C2C, idata_many_C2C, odata_many_C2C, CUFFT_FORWARD);

  //CHECK:dpct::fft::fft_engine_ptr plan_many_C2R;
  //CHECK-NEXT:int odist_many_C2R;
  //CHECK-NEXT:int ostride_many_C2R;
  //CHECK-NEXT:int * onembed_many_C2R;
  //CHECK-NEXT:int idist_many_C2R;
  //CHECK-NEXT:int istride_many_C2R;
  //CHECK-NEXT:int* inembed_many_C2R;
  //CHECK-NEXT:int * n_many_C2R;
  //CHECK-NEXT:float* odata_many_C2R;
  //CHECK-NEXT:sycl::float2* idata_many_C2R;
  //CHECK-NEXT:plan_many_C2R = dpct::fft::fft_engine::create(&q_ct1, 3, n_many_C2R, inembed_many_C2R, istride_many_C2R, idist_many_C2R, onembed_many_C2R, ostride_many_C2R, odist_many_C2R, dpct::fft::fft_type::complex_float_to_real_float, 12);
  //CHECK-NEXT:plan_many_C2R->compute<sycl::float2, float>(idata_many_C2R, odata_many_C2R, dpct::fft::fft_direction::backward);
  cufftHandle plan_many_C2R;
  int odist_many_C2R;
  int ostride_many_C2R;
  int * onembed_many_C2R;
  int idist_many_C2R;
  int istride_many_C2R;
  int* inembed_many_C2R;
  int * n_many_C2R;
  float* odata_many_C2R;
  float2* idata_many_C2R;
  cufftPlanMany(&plan_many_C2R, 3, n_many_C2R, inembed_many_C2R, istride_many_C2R, idist_many_C2R, onembed_many_C2R, ostride_many_C2R, odist_many_C2R, CUFFT_C2R, 12);
  cufftExecC2R(plan_many_C2R, idata_many_C2R, odata_many_C2R);

  //CHECK:dpct::fft::fft_engine_ptr plan_many_R2C;
  //CHECK-NEXT:int odist_many_R2C;
  //CHECK-NEXT:int ostride_many_R2C;
  //CHECK-NEXT:int * onembed_many_R2C;
  //CHECK-NEXT:int idist_many_R2C;
  //CHECK-NEXT:int istride_many_R2C;
  //CHECK-NEXT:int* inembed_many_R2C;
  //CHECK-NEXT:int * n_many_R2C;
  //CHECK-NEXT:sycl::float2* odata_many_R2C;
  //CHECK-NEXT:float* idata_many_R2C;
  //CHECK-NEXT:plan_many_R2C = dpct::fft::fft_engine::create(&q_ct1, 3, n_many_R2C, inembed_many_R2C, istride_many_R2C, idist_many_R2C, onembed_many_R2C, ostride_many_R2C, odist_many_R2C, dpct::fft::fft_type::real_float_to_complex_float, 12);
  //CHECK-NEXT:plan_many_R2C->compute<float, sycl::float2>(idata_many_R2C, odata_many_R2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_many_R2C;
  int odist_many_R2C;
  int ostride_many_R2C;
  int * onembed_many_R2C;
  int idist_many_R2C;
  int istride_many_R2C;
  int* inembed_many_R2C;
  int * n_many_R2C;
  float2* odata_many_R2C;
  float* idata_many_R2C;
  cufftPlanMany(&plan_many_R2C, 3, n_many_R2C, inembed_many_R2C, istride_many_R2C, idist_many_R2C, onembed_many_R2C, ostride_many_R2C, odist_many_R2C, CUFFT_R2C, 12);
  cufftExecR2C(plan_many_R2C, idata_many_R2C, odata_many_R2C);

  //CHECK:dpct::fft::fft_engine_ptr plan_many_Z2Z;
  //CHECK-NEXT:int odist_many_Z2Z;
  //CHECK-NEXT:int ostride_many_Z2Z;
  //CHECK-NEXT:int * onembed_many_Z2Z;
  //CHECK-NEXT:int idist_many_Z2Z;
  //CHECK-NEXT:int istride_many_Z2Z;
  //CHECK-NEXT:int* inembed_many_Z2Z;
  //CHECK-NEXT:int * n_many_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_many_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_many_Z2Z;
  //CHECK-NEXT:plan_many_Z2Z = dpct::fft::fft_engine::create(&q_ct1, 3, n_many_Z2Z, inembed_many_Z2Z, istride_many_Z2Z, idist_many_Z2Z, onembed_many_Z2Z, ostride_many_Z2Z, odist_many_Z2Z, dpct::fft::fft_type::complex_double_to_complex_double, 12);
  //CHECK-NEXT:plan_many_Z2Z->compute<sycl::double2, sycl::double2>(idata_many_Z2Z, odata_many_Z2Z, dpct::fft::fft_direction::backward);
  cufftHandle plan_many_Z2Z;
  int odist_many_Z2Z;
  int ostride_many_Z2Z;
  int * onembed_many_Z2Z;
  int idist_many_Z2Z;
  int istride_many_Z2Z;
  int* inembed_many_Z2Z;
  int * n_many_Z2Z;
  double2* odata_many_Z2Z;
  double2* idata_many_Z2Z;
  cufftPlanMany(&plan_many_Z2Z, 3, n_many_Z2Z, inembed_many_Z2Z, istride_many_Z2Z, idist_many_Z2Z, onembed_many_Z2Z, ostride_many_Z2Z, odist_many_Z2Z, CUFFT_Z2Z, 12);
  cufftExecZ2Z(plan_many_Z2Z, idata_many_Z2Z, odata_many_Z2Z, CUFFT_INVERSE);

  //CHECK:dpct::fft::fft_engine_ptr plan_many_Z2D;
  //CHECK-NEXT:int odist_many_Z2D;
  //CHECK-NEXT:int ostride_many_Z2D;
  //CHECK-NEXT:int * onembed_many_Z2D;
  //CHECK-NEXT:int idist_many_Z2D;
  //CHECK-NEXT:int istride_many_Z2D;
  //CHECK-NEXT:int* inembed_many_Z2D;
  //CHECK-NEXT:int * n_many_Z2D;
  //CHECK-NEXT:double* odata_many_Z2D;
  //CHECK-NEXT:sycl::double2* idata_many_Z2D;
  //CHECK-NEXT:plan_many_Z2D = dpct::fft::fft_engine::create(&q_ct1, 3, n_many_Z2D, inembed_many_Z2D, istride_many_Z2D, idist_many_Z2D, onembed_many_Z2D, ostride_many_Z2D, odist_many_Z2D, dpct::fft::fft_type::complex_double_to_real_double, 12);
  //CHECK-NEXT:plan_many_Z2D->compute<sycl::double2, double>(idata_many_Z2D, odata_many_Z2D, dpct::fft::fft_direction::backward);
  cufftHandle plan_many_Z2D;
  int odist_many_Z2D;
  int ostride_many_Z2D;
  int * onembed_many_Z2D;
  int idist_many_Z2D;
  int istride_many_Z2D;
  int* inembed_many_Z2D;
  int * n_many_Z2D;
  double* odata_many_Z2D;
  double2* idata_many_Z2D;
  cufftPlanMany(&plan_many_Z2D, 3, n_many_Z2D, inembed_many_Z2D, istride_many_Z2D, idist_many_Z2D, onembed_many_Z2D, ostride_many_Z2D, odist_many_Z2D, CUFFT_Z2D, 12);
  cufftExecZ2D(plan_many_Z2D, idata_many_Z2D, odata_many_Z2D);

  //CHECK:dpct::fft::fft_engine_ptr plan_many_D2Z;
  //CHECK-NEXT:int odist_many_D2Z;
  //CHECK-NEXT:int ostride_many_D2Z;
  //CHECK-NEXT:int * onembed_many_D2Z;
  //CHECK-NEXT:int idist_many_D2Z;
  //CHECK-NEXT:int istride_many_D2Z;
  //CHECK-NEXT:int* inembed_many_D2Z;
  //CHECK-NEXT:int * n_many_D2Z;
  //CHECK-NEXT:sycl::double2* odata_many_D2Z;
  //CHECK-NEXT:double* idata_many_D2Z;
  //CHECK-NEXT:plan_many_D2Z = dpct::fft::fft_engine::create(&q_ct1, 3, n_many_D2Z, inembed_many_D2Z, istride_many_D2Z, idist_many_D2Z, onembed_many_D2Z, ostride_many_D2Z, odist_many_D2Z, dpct::fft::fft_type::real_double_to_complex_double, 12);
  //CHECK-NEXT:plan_many_D2Z->compute<double, sycl::double2>(idata_many_D2Z, odata_many_D2Z, dpct::fft::fft_direction::forward);
  cufftHandle plan_many_D2Z;
  int odist_many_D2Z;
  int ostride_many_D2Z;
  int * onembed_many_D2Z;
  int idist_many_D2Z;
  int istride_many_D2Z;
  int* inembed_many_D2Z;
  int * n_many_D2Z;
  double2* odata_many_D2Z;
  double* idata_many_D2Z;
  cufftPlanMany(&plan_many_D2Z, 3, n_many_D2Z, inembed_many_D2Z, istride_many_D2Z, idist_many_D2Z, onembed_many_D2Z, ostride_many_D2Z, odist_many_D2Z, CUFFT_D2Z, 12);
  cufftExecD2Z(plan_many_D2Z, idata_many_D2Z, odata_many_D2Z);

  size_t* work_size;
  //CHECK:dpct::fft::fft_engine_ptr plan_m1d_C2C;
  //CHECK-NEXT:sycl::float2* odata_m1d_C2C;
  //CHECK-NEXT:sycl::float2* idata_m1d_C2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_C2C->commit(&q_ct1, 10, dpct::fft::fft_type::complex_float_to_complex_float, 3, work_size);
  //CHECK-NEXT:plan_m1d_C2C->compute<sycl::float2, sycl::float2>(idata_m1d_C2C, odata_m1d_C2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_m1d_C2C;
  float2* odata_m1d_C2C;
  float2* idata_m1d_C2C;
  cufftMakePlan1d(plan_m1d_C2C, 10, CUFFT_C2C, 3, work_size);
  cufftExecC2C(plan_m1d_C2C, idata_m1d_C2C, odata_m1d_C2C, CUFFT_FORWARD);

  //CHECK:dpct::fft::fft_engine_ptr plan_m1d_C2R;
  //CHECK-NEXT:float* odata_m1d_C2R;
  //CHECK-NEXT:sycl::float2* idata_m1d_C2R;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_C2R->commit(&q_ct1, 10, dpct::fft::fft_type::complex_float_to_real_float, 3, work_size);
  //CHECK-NEXT:plan_m1d_C2R->compute<sycl::float2, float>(idata_m1d_C2R, odata_m1d_C2R, dpct::fft::fft_direction::backward);
  cufftHandle plan_m1d_C2R;
  float* odata_m1d_C2R;
  float2* idata_m1d_C2R;
  cufftMakePlan1d(plan_m1d_C2R, 10, CUFFT_C2R, 3, work_size);
  cufftExecC2R(plan_m1d_C2R, idata_m1d_C2R, odata_m1d_C2R);

  //CHECK:dpct::fft::fft_engine_ptr plan_m1d_R2C;
  //CHECK-NEXT:sycl::float2* odata_m1d_R2C;
  //CHECK-NEXT:float* idata_m1d_R2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_R2C->commit(&q_ct1, 10, dpct::fft::fft_type::real_float_to_complex_float, 3, work_size);
  //CHECK-NEXT:plan_m1d_R2C->compute<float, sycl::float2>(idata_m1d_R2C, odata_m1d_R2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_m1d_R2C;
  float2* odata_m1d_R2C;
  float* idata_m1d_R2C;
  cufftMakePlan1d(plan_m1d_R2C, 10, CUFFT_R2C, 3, work_size);
  cufftExecR2C(plan_m1d_R2C, idata_m1d_R2C, odata_m1d_R2C);

  //CHECK:dpct::fft::fft_engine_ptr plan_m1d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_m1d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_m1d_Z2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_Z2Z->commit(&q_ct1, 10, dpct::fft::fft_type::complex_double_to_complex_double, 3, work_size);
  //CHECK-NEXT:plan_m1d_Z2Z->compute<sycl::double2, sycl::double2>(idata_m1d_Z2Z, odata_m1d_Z2Z, dpct::fft::fft_direction::backward);
  cufftHandle plan_m1d_Z2Z;
  double2* odata_m1d_Z2Z;
  double2* idata_m1d_Z2Z;
  cufftMakePlan1d(plan_m1d_Z2Z, 10, CUFFT_Z2Z, 3, work_size);
  cufftExecZ2Z(plan_m1d_Z2Z, idata_m1d_Z2Z, odata_m1d_Z2Z, CUFFT_INVERSE);

  //CHECK:dpct::fft::fft_engine_ptr plan_m1d_Z2D;
  //CHECK-NEXT:double* odata_m1d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_m1d_Z2D;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_Z2D->commit(&q_ct1, 10, dpct::fft::fft_type::complex_double_to_real_double, 3, work_size);
  //CHECK-NEXT:plan_m1d_Z2D->compute<sycl::double2, double>(idata_m1d_Z2D, odata_m1d_Z2D, dpct::fft::fft_direction::backward);
  cufftHandle plan_m1d_Z2D;
  double* odata_m1d_Z2D;
  double2* idata_m1d_Z2D;
  cufftMakePlan1d(plan_m1d_Z2D, 10, CUFFT_Z2D, 3, work_size);
  cufftExecZ2D(plan_m1d_Z2D, idata_m1d_Z2D, odata_m1d_Z2D);

  //CHECK:dpct::fft::fft_engine_ptr plan_m1d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_m1d_D2Z;
  //CHECK-NEXT:double* idata_m1d_D2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_D2Z->commit(&q_ct1, 10, dpct::fft::fft_type::real_double_to_complex_double, 3, work_size);
  //CHECK-NEXT:plan_m1d_D2Z->compute<double, sycl::double2>(idata_m1d_D2Z, odata_m1d_D2Z, dpct::fft::fft_direction::forward);
  cufftHandle plan_m1d_D2Z;
  double2* odata_m1d_D2Z;
  double* idata_m1d_D2Z;
  cufftMakePlan1d(plan_m1d_D2Z, 10, CUFFT_D2Z, 3, work_size);
  cufftExecD2Z(plan_m1d_D2Z, idata_m1d_D2Z, odata_m1d_D2Z);

  //CHECK:dpct::fft::fft_engine_ptr plan_m2d_C2C;
  //CHECK-NEXT:sycl::float2* odata_m2d_C2C;
  //CHECK-NEXT:sycl::float2* idata_m2d_C2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_C2C->commit(&q_ct1, 10, 20, dpct::fft::fft_type::complex_float_to_complex_float, work_size);
  //CHECK-NEXT:plan_m2d_C2C->compute<sycl::float2, sycl::float2>(idata_m2d_C2C, odata_m2d_C2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_m2d_C2C;
  float2* odata_m2d_C2C;
  float2* idata_m2d_C2C;
  cufftMakePlan2d(plan_m2d_C2C, 10, 20, CUFFT_C2C, work_size);
  cufftExecC2C(plan_m2d_C2C, idata_m2d_C2C, odata_m2d_C2C, CUFFT_FORWARD);

  //CHECK:dpct::fft::fft_engine_ptr plan_m2d_C2R;
  //CHECK-NEXT:float* odata_m2d_C2R;
  //CHECK-NEXT:sycl::float2* idata_m2d_C2R;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_C2R->commit(&q_ct1, 10, 20, dpct::fft::fft_type::complex_float_to_real_float, work_size);
  //CHECK-NEXT:plan_m2d_C2R->compute<sycl::float2, float>(idata_m2d_C2R, odata_m2d_C2R, dpct::fft::fft_direction::backward);
  cufftHandle plan_m2d_C2R;
  float* odata_m2d_C2R;
  float2* idata_m2d_C2R;
  cufftMakePlan2d(plan_m2d_C2R, 10, 20, CUFFT_C2R, work_size);
  cufftExecC2R(plan_m2d_C2R, idata_m2d_C2R, odata_m2d_C2R);

  //CHECK:dpct::fft::fft_engine_ptr plan_m2d_R2C;
  //CHECK-NEXT:sycl::float2* odata_m2d_R2C;
  //CHECK-NEXT:float* idata_m2d_R2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_R2C->commit(&q_ct1, 10, 20, dpct::fft::fft_type::real_float_to_complex_float, work_size);
  //CHECK-NEXT:plan_m2d_R2C->compute<float, sycl::float2>(idata_m2d_R2C, odata_m2d_R2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_m2d_R2C;
  float2* odata_m2d_R2C;
  float* idata_m2d_R2C;
  cufftMakePlan2d(plan_m2d_R2C, 10, 20, CUFFT_R2C, work_size);
  cufftExecR2C(plan_m2d_R2C, idata_m2d_R2C, odata_m2d_R2C);

  //CHECK:dpct::fft::fft_engine_ptr plan_m2d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_m2d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_m2d_Z2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_Z2Z->commit(&q_ct1, 10, 20, dpct::fft::fft_type::complex_double_to_complex_double, work_size);
  //CHECK-NEXT:plan_m2d_Z2Z->compute<sycl::double2, sycl::double2>(idata_m2d_Z2Z, odata_m2d_Z2Z, dpct::fft::fft_direction::backward);
  cufftHandle plan_m2d_Z2Z;
  double2* odata_m2d_Z2Z;
  double2* idata_m2d_Z2Z;
  cufftMakePlan2d(plan_m2d_Z2Z, 10, 20, CUFFT_Z2Z, work_size);
  cufftExecZ2Z(plan_m2d_Z2Z, idata_m2d_Z2Z, odata_m2d_Z2Z, CUFFT_INVERSE);

  //CHECK:dpct::fft::fft_engine_ptr plan_m2d_Z2D;
  //CHECK-NEXT:double* odata_m2d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_m2d_Z2D;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_Z2D->commit(&q_ct1, 10, 20, dpct::fft::fft_type::complex_double_to_real_double, work_size);
  //CHECK-NEXT:plan_m2d_Z2D->compute<sycl::double2, double>(idata_m2d_Z2D, odata_m2d_Z2D, dpct::fft::fft_direction::backward);
  cufftHandle plan_m2d_Z2D;
  double* odata_m2d_Z2D;
  double2* idata_m2d_Z2D;
  cufftMakePlan2d(plan_m2d_Z2D, 10, 20, CUFFT_Z2D, work_size);
  cufftExecZ2D(plan_m2d_Z2D, idata_m2d_Z2D, odata_m2d_Z2D);

  //CHECK:dpct::fft::fft_engine_ptr plan_m2d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_m2d_D2Z;
  //CHECK-NEXT:double* idata_m2d_D2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_D2Z->commit(&q_ct1, 10, 20, dpct::fft::fft_type::real_double_to_complex_double, work_size);
  //CHECK-NEXT:plan_m2d_D2Z->compute<double, sycl::double2>(idata_m2d_D2Z, odata_m2d_D2Z, dpct::fft::fft_direction::forward);
  cufftHandle plan_m2d_D2Z;
  double2* odata_m2d_D2Z;
  double* idata_m2d_D2Z;
  cufftMakePlan2d(plan_m2d_D2Z, 10, 20, CUFFT_D2Z, work_size);
  cufftExecD2Z(plan_m2d_D2Z, idata_m2d_D2Z, odata_m2d_D2Z);

  //CHECK:dpct::fft::fft_engine_ptr plan_m3d_C2C;
  //CHECK-NEXT:sycl::float2* odata_m3d_C2C;
  //CHECK-NEXT:sycl::float2* idata_m3d_C2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_C2C->commit(&q_ct1, 10, 20, 30, dpct::fft::fft_type::complex_float_to_complex_float, work_size);
  //CHECK-NEXT:plan_m3d_C2C->compute<sycl::float2, sycl::float2>(idata_m3d_C2C, odata_m3d_C2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_m3d_C2C;
  float2* odata_m3d_C2C;
  float2* idata_m3d_C2C;
  cufftMakePlan3d(plan_m3d_C2C, 10, 20, 30, CUFFT_C2C, work_size);
  cufftExecC2C(plan_m3d_C2C, idata_m3d_C2C, odata_m3d_C2C, CUFFT_FORWARD);

  //CHECK:dpct::fft::fft_engine_ptr plan_m3d_C2R;
  //CHECK-NEXT:float* odata_m3d_C2R;
  //CHECK-NEXT:sycl::float2* idata_m3d_C2R;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_C2R->commit(&q_ct1, 10, 20, 30, dpct::fft::fft_type::complex_float_to_real_float, work_size);
  //CHECK-NEXT:plan_m3d_C2R->compute<sycl::float2, float>(idata_m3d_C2R, odata_m3d_C2R, dpct::fft::fft_direction::backward);
  cufftHandle plan_m3d_C2R;
  float* odata_m3d_C2R;
  float2* idata_m3d_C2R;
  cufftMakePlan3d(plan_m3d_C2R, 10, 20, 30, CUFFT_C2R, work_size);
  cufftExecC2R(plan_m3d_C2R, idata_m3d_C2R, odata_m3d_C2R);

  //CHECK:dpct::fft::fft_engine_ptr plan_m3d_R2C;
  //CHECK-NEXT:sycl::float2* odata_m3d_R2C;
  //CHECK-NEXT:float* idata_m3d_R2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_R2C->commit(&q_ct1, 10, 20, 30, dpct::fft::fft_type::real_float_to_complex_float, work_size);
  //CHECK-NEXT:plan_m3d_R2C->compute<float, sycl::float2>(idata_m3d_R2C, odata_m3d_R2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_m3d_R2C;
  float2* odata_m3d_R2C;
  float* idata_m3d_R2C;
  cufftMakePlan3d(plan_m3d_R2C, 10, 20, 30, CUFFT_R2C, work_size);
  cufftExecR2C(plan_m3d_R2C, idata_m3d_R2C, odata_m3d_R2C);

  //CHECK:dpct::fft::fft_engine_ptr plan_m3d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_m3d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_m3d_Z2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_Z2Z->commit(&q_ct1, 10, 20, 30, dpct::fft::fft_type::complex_double_to_complex_double, work_size);
  //CHECK-NEXT:plan_m3d_Z2Z->compute<sycl::double2, sycl::double2>(idata_m3d_Z2Z, odata_m3d_Z2Z, dpct::fft::fft_direction::backward);
  cufftHandle plan_m3d_Z2Z;
  double2* odata_m3d_Z2Z;
  double2* idata_m3d_Z2Z;
  cufftMakePlan3d(plan_m3d_Z2Z, 10, 20, 30, CUFFT_Z2Z, work_size);
  cufftExecZ2Z(plan_m3d_Z2Z, idata_m3d_Z2Z, odata_m3d_Z2Z, CUFFT_INVERSE);

  //CHECK:dpct::fft::fft_engine_ptr plan_m3d_Z2D;
  //CHECK-NEXT:double* odata_m3d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_m3d_Z2D;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_Z2D->commit(&q_ct1, 10, 20, 30, dpct::fft::fft_type::complex_double_to_real_double, work_size);
  //CHECK-NEXT:plan_m3d_Z2D->compute<sycl::double2, double>(idata_m3d_Z2D, odata_m3d_Z2D, dpct::fft::fft_direction::backward);
  cufftHandle plan_m3d_Z2D;
  double* odata_m3d_Z2D;
  double2* idata_m3d_Z2D;
  cufftMakePlan3d(plan_m3d_Z2D, 10, 20, 30, CUFFT_Z2D, work_size);
  cufftExecZ2D(plan_m3d_Z2D, idata_m3d_Z2D, odata_m3d_Z2D);

  //CHECK:dpct::fft::fft_engine_ptr plan_m3d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_m3d_D2Z;
  //CHECK-NEXT:double* idata_m3d_D2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_D2Z->commit(&q_ct1, 10, 20, 30, dpct::fft::fft_type::real_double_to_complex_double, work_size);
  //CHECK-NEXT:plan_m3d_D2Z->compute<double, sycl::double2>(idata_m3d_D2Z, odata_m3d_D2Z, dpct::fft::fft_direction::forward);
  cufftHandle plan_m3d_D2Z;
  double2* odata_m3d_D2Z;
  double* idata_m3d_D2Z;
  cufftMakePlan3d(plan_m3d_D2Z, 10, 20, 30, CUFFT_D2Z, work_size);
  cufftExecD2Z(plan_m3d_D2Z, idata_m3d_D2Z, odata_m3d_D2Z);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany_C2C;
  //CHECK-NEXT:int odist_mmany_C2C;
  //CHECK-NEXT:int ostride_mmany_C2C;
  //CHECK-NEXT:int * onembed_mmany_C2C;
  //CHECK-NEXT:int idist_mmany_C2C;
  //CHECK-NEXT:int istride_mmany_C2C;
  //CHECK-NEXT:int* inembed_mmany_C2C;
  //CHECK-NEXT:int * n_mmany_C2C;
  //CHECK-NEXT:sycl::float2* odata_mmany_C2C;
  //CHECK-NEXT:sycl::float2* idata_mmany_C2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_C2C->commit(&q_ct1, 3, n_mmany_C2C, inembed_mmany_C2C, istride_mmany_C2C, idist_mmany_C2C, onembed_mmany_C2C, ostride_mmany_C2C, odist_mmany_C2C, dpct::fft::fft_type::complex_float_to_complex_float, 12, work_size);
  //CHECK-NEXT:plan_mmany_C2C->compute<sycl::float2, sycl::float2>(idata_mmany_C2C, odata_mmany_C2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_mmany_C2C;
  int odist_mmany_C2C;
  int ostride_mmany_C2C;
  int * onembed_mmany_C2C;
  int idist_mmany_C2C;
  int istride_mmany_C2C;
  int* inembed_mmany_C2C;
  int * n_mmany_C2C;
  float2* odata_mmany_C2C;
  float2* idata_mmany_C2C;
  cufftMakePlanMany(plan_mmany_C2C, 3, n_mmany_C2C, inembed_mmany_C2C, istride_mmany_C2C, idist_mmany_C2C, onembed_mmany_C2C, ostride_mmany_C2C, odist_mmany_C2C, CUFFT_C2C, 12, work_size);
  cufftExecC2C(plan_mmany_C2C, idata_mmany_C2C, odata_mmany_C2C, CUFFT_FORWARD);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany_C2R;
  //CHECK-NEXT:int odist_mmany_C2R;
  //CHECK-NEXT:int ostride_mmany_C2R;
  //CHECK-NEXT:int * onembed_mmany_C2R;
  //CHECK-NEXT:int idist_mmany_C2R;
  //CHECK-NEXT:int istride_mmany_C2R;
  //CHECK-NEXT:int* inembed_mmany_C2R;
  //CHECK-NEXT:int * n_mmany_C2R;
  //CHECK-NEXT:float* odata_mmany_C2R;
  //CHECK-NEXT:sycl::float2* idata_mmany_C2R;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_C2R->commit(&q_ct1, 3, n_mmany_C2R, inembed_mmany_C2R, istride_mmany_C2R, idist_mmany_C2R, onembed_mmany_C2R, ostride_mmany_C2R, odist_mmany_C2R, dpct::fft::fft_type::complex_float_to_real_float, 12, work_size);
  //CHECK-NEXT:plan_mmany_C2R->compute<sycl::float2, float>(idata_mmany_C2R, odata_mmany_C2R, dpct::fft::fft_direction::backward);
  cufftHandle plan_mmany_C2R;
  int odist_mmany_C2R;
  int ostride_mmany_C2R;
  int * onembed_mmany_C2R;
  int idist_mmany_C2R;
  int istride_mmany_C2R;
  int* inembed_mmany_C2R;
  int * n_mmany_C2R;
  float* odata_mmany_C2R;
  float2* idata_mmany_C2R;
  cufftMakePlanMany(plan_mmany_C2R, 3, n_mmany_C2R, inembed_mmany_C2R, istride_mmany_C2R, idist_mmany_C2R, onembed_mmany_C2R, ostride_mmany_C2R, odist_mmany_C2R, CUFFT_C2R, 12, work_size);
  cufftExecC2R(plan_mmany_C2R, idata_mmany_C2R, odata_mmany_C2R);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany_R2C;
  //CHECK-NEXT:int odist_mmany_R2C;
  //CHECK-NEXT:int ostride_mmany_R2C;
  //CHECK-NEXT:int * onembed_mmany_R2C;
  //CHECK-NEXT:int idist_mmany_R2C;
  //CHECK-NEXT:int istride_mmany_R2C;
  //CHECK-NEXT:int* inembed_mmany_R2C;
  //CHECK-NEXT:int * n_mmany_R2C;
  //CHECK-NEXT:sycl::float2* odata_mmany_R2C;
  //CHECK-NEXT:float* idata_mmany_R2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_R2C->commit(&q_ct1, 3, n_mmany_R2C, inembed_mmany_R2C, istride_mmany_R2C, idist_mmany_R2C, onembed_mmany_R2C, ostride_mmany_R2C, odist_mmany_R2C, dpct::fft::fft_type::real_float_to_complex_float, 12, work_size);
  //CHECK-NEXT:plan_mmany_R2C->compute<float, sycl::float2>(idata_mmany_R2C, odata_mmany_R2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_mmany_R2C;
  int odist_mmany_R2C;
  int ostride_mmany_R2C;
  int * onembed_mmany_R2C;
  int idist_mmany_R2C;
  int istride_mmany_R2C;
  int* inembed_mmany_R2C;
  int * n_mmany_R2C;
  float2* odata_mmany_R2C;
  float* idata_mmany_R2C;
  cufftMakePlanMany(plan_mmany_R2C, 3, n_mmany_R2C, inembed_mmany_R2C, istride_mmany_R2C, idist_mmany_R2C, onembed_mmany_R2C, ostride_mmany_R2C, odist_mmany_R2C, CUFFT_R2C, 12, work_size);
  cufftExecR2C(plan_mmany_R2C, idata_mmany_R2C, odata_mmany_R2C);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany_Z2Z;
  //CHECK-NEXT:int odist_mmany_Z2Z;
  //CHECK-NEXT:int ostride_mmany_Z2Z;
  //CHECK-NEXT:int * onembed_mmany_Z2Z;
  //CHECK-NEXT:int idist_mmany_Z2Z;
  //CHECK-NEXT:int istride_mmany_Z2Z;
  //CHECK-NEXT:int* inembed_mmany_Z2Z;
  //CHECK-NEXT:int * n_mmany_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_mmany_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_mmany_Z2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_Z2Z->commit(&q_ct1, 3, n_mmany_Z2Z, inembed_mmany_Z2Z, istride_mmany_Z2Z, idist_mmany_Z2Z, onembed_mmany_Z2Z, ostride_mmany_Z2Z, odist_mmany_Z2Z, dpct::fft::fft_type::complex_double_to_complex_double, 12, work_size);
  //CHECK-NEXT:plan_mmany_Z2Z->compute<sycl::double2, sycl::double2>(idata_mmany_Z2Z, odata_mmany_Z2Z, dpct::fft::fft_direction::backward);
  cufftHandle plan_mmany_Z2Z;
  int odist_mmany_Z2Z;
  int ostride_mmany_Z2Z;
  int * onembed_mmany_Z2Z;
  int idist_mmany_Z2Z;
  int istride_mmany_Z2Z;
  int* inembed_mmany_Z2Z;
  int * n_mmany_Z2Z;
  double2* odata_mmany_Z2Z;
  double2* idata_mmany_Z2Z;
  cufftMakePlanMany(plan_mmany_Z2Z, 3, n_mmany_Z2Z, inembed_mmany_Z2Z, istride_mmany_Z2Z, idist_mmany_Z2Z, onembed_mmany_Z2Z, ostride_mmany_Z2Z, odist_mmany_Z2Z, CUFFT_Z2Z, 12, work_size);
  cufftExecZ2Z(plan_mmany_Z2Z, idata_mmany_Z2Z, odata_mmany_Z2Z, CUFFT_INVERSE);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany_Z2D;
  //CHECK-NEXT:int odist_mmany_Z2D;
  //CHECK-NEXT:int ostride_mmany_Z2D;
  //CHECK-NEXT:int * onembed_mmany_Z2D;
  //CHECK-NEXT:int idist_mmany_Z2D;
  //CHECK-NEXT:int istride_mmany_Z2D;
  //CHECK-NEXT:int* inembed_mmany_Z2D;
  //CHECK-NEXT:int * n_mmany_Z2D;
  //CHECK-NEXT:double* odata_mmany_Z2D;
  //CHECK-NEXT:sycl::double2* idata_mmany_Z2D;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_Z2D->commit(&q_ct1, 3, n_mmany_Z2D, inembed_mmany_Z2D, istride_mmany_Z2D, idist_mmany_Z2D, onembed_mmany_Z2D, ostride_mmany_Z2D, odist_mmany_Z2D, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size);
  //CHECK-NEXT:plan_mmany_Z2D->compute<sycl::double2, double>(idata_mmany_Z2D, odata_mmany_Z2D, dpct::fft::fft_direction::backward);
  cufftHandle plan_mmany_Z2D;
  int odist_mmany_Z2D;
  int ostride_mmany_Z2D;
  int * onembed_mmany_Z2D;
  int idist_mmany_Z2D;
  int istride_mmany_Z2D;
  int* inembed_mmany_Z2D;
  int * n_mmany_Z2D;
  double* odata_mmany_Z2D;
  double2* idata_mmany_Z2D;
  cufftMakePlanMany(plan_mmany_Z2D, 3, n_mmany_Z2D, inembed_mmany_Z2D, istride_mmany_Z2D, idist_mmany_Z2D, onembed_mmany_Z2D, ostride_mmany_Z2D, odist_mmany_Z2D, CUFFT_Z2D, 12, work_size);
  cufftExecZ2D(plan_mmany_Z2D, idata_mmany_Z2D, odata_mmany_Z2D);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany_D2Z;
  //CHECK-NEXT:int odist_mmany_D2Z;
  //CHECK-NEXT:int ostride_mmany_D2Z;
  //CHECK-NEXT:int * onembed_mmany_D2Z;
  //CHECK-NEXT:int idist_mmany_D2Z;
  //CHECK-NEXT:int istride_mmany_D2Z;
  //CHECK-NEXT:int* inembed_mmany_D2Z;
  //CHECK-NEXT:int * n_mmany_D2Z;
  //CHECK-NEXT:sycl::double2* odata_mmany_D2Z;
  //CHECK-NEXT:double* idata_mmany_D2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_D2Z->commit(&q_ct1, 3, n_mmany_D2Z, inembed_mmany_D2Z, istride_mmany_D2Z, idist_mmany_D2Z, onembed_mmany_D2Z, ostride_mmany_D2Z, odist_mmany_D2Z, dpct::fft::fft_type::real_double_to_complex_double, 12, work_size);
  //CHECK-NEXT:plan_mmany_D2Z->compute<double, sycl::double2>(idata_mmany_D2Z, odata_mmany_D2Z, dpct::fft::fft_direction::forward);
  cufftHandle plan_mmany_D2Z;
  int odist_mmany_D2Z;
  int ostride_mmany_D2Z;
  int * onembed_mmany_D2Z;
  int idist_mmany_D2Z;
  int istride_mmany_D2Z;
  int* inembed_mmany_D2Z;
  int * n_mmany_D2Z;
  double2* odata_mmany_D2Z;
  double* idata_mmany_D2Z;
  cufftMakePlanMany(plan_mmany_D2Z, 3, n_mmany_D2Z, inembed_mmany_D2Z, istride_mmany_D2Z, idist_mmany_D2Z, onembed_mmany_D2Z, ostride_mmany_D2Z, odist_mmany_D2Z, CUFFT_D2Z, 12, work_size);
  cufftExecD2Z(plan_mmany_D2Z, idata_mmany_D2Z, odata_mmany_D2Z);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany64_C2C;
  //CHECK-NEXT:long long int odist_mmany64_C2C;
  //CHECK-NEXT:long long int ostride_mmany64_C2C;
  //CHECK-NEXT:long long int * onembed_mmany64_C2C;
  //CHECK-NEXT:long long int idist_mmany64_C2C;
  //CHECK-NEXT:long long int istride_mmany64_C2C;
  //CHECK-NEXT:long long int* inembed_mmany64_C2C;
  //CHECK-NEXT:long long int * n_mmany64_C2C;
  //CHECK-NEXT:sycl::float2* odata_mmany64_C2C;
  //CHECK-NEXT:sycl::float2* idata_mmany64_C2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_C2C->commit(&q_ct1, 3, n_mmany64_C2C, inembed_mmany64_C2C, istride_mmany64_C2C, idist_mmany64_C2C, onembed_mmany64_C2C, ostride_mmany64_C2C, odist_mmany64_C2C, dpct::fft::fft_type::complex_float_to_complex_float, 12, work_size);
  //CHECK-NEXT:plan_mmany64_C2C->compute<sycl::float2, sycl::float2>(idata_mmany64_C2C, odata_mmany64_C2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_mmany64_C2C;
  long long int odist_mmany64_C2C;
  long long int ostride_mmany64_C2C;
  long long int * onembed_mmany64_C2C;
  long long int idist_mmany64_C2C;
  long long int istride_mmany64_C2C;
  long long int* inembed_mmany64_C2C;
  long long int * n_mmany64_C2C;
  float2* odata_mmany64_C2C;
  float2* idata_mmany64_C2C;
  cufftMakePlanMany64(plan_mmany64_C2C, 3, n_mmany64_C2C, inembed_mmany64_C2C, istride_mmany64_C2C, idist_mmany64_C2C, onembed_mmany64_C2C, ostride_mmany64_C2C, odist_mmany64_C2C, CUFFT_C2C, 12, work_size);
  cufftExecC2C(plan_mmany64_C2C, idata_mmany64_C2C, odata_mmany64_C2C, CUFFT_FORWARD);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany64_C2R;
  //CHECK-NEXT:long long int odist_mmany64_C2R;
  //CHECK-NEXT:long long int ostride_mmany64_C2R;
  //CHECK-NEXT:long long int * onembed_mmany64_C2R;
  //CHECK-NEXT:long long int idist_mmany64_C2R;
  //CHECK-NEXT:long long int istride_mmany64_C2R;
  //CHECK-NEXT:long long int* inembed_mmany64_C2R;
  //CHECK-NEXT:long long int * n_mmany64_C2R;
  //CHECK-NEXT:float* odata_mmany64_C2R;
  //CHECK-NEXT:sycl::float2* idata_mmany64_C2R;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_C2R->commit(&q_ct1, 3, n_mmany64_C2R, inembed_mmany64_C2R, istride_mmany64_C2R, idist_mmany64_C2R, onembed_mmany64_C2R, ostride_mmany64_C2R, odist_mmany64_C2R, dpct::fft::fft_type::complex_float_to_real_float, 12, work_size);
  //CHECK-NEXT:plan_mmany64_C2R->compute<sycl::float2, float>(idata_mmany64_C2R, odata_mmany64_C2R, dpct::fft::fft_direction::backward);
  cufftHandle plan_mmany64_C2R;
  long long int odist_mmany64_C2R;
  long long int ostride_mmany64_C2R;
  long long int * onembed_mmany64_C2R;
  long long int idist_mmany64_C2R;
  long long int istride_mmany64_C2R;
  long long int* inembed_mmany64_C2R;
  long long int * n_mmany64_C2R;
  float* odata_mmany64_C2R;
  float2* idata_mmany64_C2R;
  cufftMakePlanMany64(plan_mmany64_C2R, 3, n_mmany64_C2R, inembed_mmany64_C2R, istride_mmany64_C2R, idist_mmany64_C2R, onembed_mmany64_C2R, ostride_mmany64_C2R, odist_mmany64_C2R, CUFFT_C2R, 12, work_size);
  cufftExecC2R(plan_mmany64_C2R, idata_mmany64_C2R, odata_mmany64_C2R);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany64_R2C;
  //CHECK-NEXT:long long int odist_mmany64_R2C;
  //CHECK-NEXT:long long int ostride_mmany64_R2C;
  //CHECK-NEXT:long long int * onembed_mmany64_R2C;
  //CHECK-NEXT:long long int idist_mmany64_R2C;
  //CHECK-NEXT:long long int istride_mmany64_R2C;
  //CHECK-NEXT:long long int* inembed_mmany64_R2C;
  //CHECK-NEXT:long long int * n_mmany64_R2C;
  //CHECK-NEXT:sycl::float2* odata_mmany64_R2C;
  //CHECK-NEXT:float* idata_mmany64_R2C;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_R2C->commit(&q_ct1, 3, n_mmany64_R2C, inembed_mmany64_R2C, istride_mmany64_R2C, idist_mmany64_R2C, onembed_mmany64_R2C, ostride_mmany64_R2C, odist_mmany64_R2C, dpct::fft::fft_type::real_float_to_complex_float, 12, work_size);
  //CHECK-NEXT:plan_mmany64_R2C->compute<float, sycl::float2>(idata_mmany64_R2C, odata_mmany64_R2C, dpct::fft::fft_direction::forward);
  cufftHandle plan_mmany64_R2C;
  long long int odist_mmany64_R2C;
  long long int ostride_mmany64_R2C;
  long long int * onembed_mmany64_R2C;
  long long int idist_mmany64_R2C;
  long long int istride_mmany64_R2C;
  long long int* inembed_mmany64_R2C;
  long long int * n_mmany64_R2C;
  float2* odata_mmany64_R2C;
  float* idata_mmany64_R2C;
  cufftMakePlanMany64(plan_mmany64_R2C, 3, n_mmany64_R2C, inembed_mmany64_R2C, istride_mmany64_R2C, idist_mmany64_R2C, onembed_mmany64_R2C, ostride_mmany64_R2C, odist_mmany64_R2C, CUFFT_R2C, 12, work_size);
  cufftExecR2C(plan_mmany64_R2C, idata_mmany64_R2C, odata_mmany64_R2C);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany64_Z2Z;
  //CHECK-NEXT:long long int odist_mmany64_Z2Z;
  //CHECK-NEXT:long long int ostride_mmany64_Z2Z;
  //CHECK-NEXT:long long int * onembed_mmany64_Z2Z;
  //CHECK-NEXT:long long int idist_mmany64_Z2Z;
  //CHECK-NEXT:long long int istride_mmany64_Z2Z;
  //CHECK-NEXT:long long int* inembed_mmany64_Z2Z;
  //CHECK-NEXT:long long int * n_mmany64_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_mmany64_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_mmany64_Z2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_Z2Z->commit(&q_ct1, 3, n_mmany64_Z2Z, inembed_mmany64_Z2Z, istride_mmany64_Z2Z, idist_mmany64_Z2Z, onembed_mmany64_Z2Z, ostride_mmany64_Z2Z, odist_mmany64_Z2Z, dpct::fft::fft_type::complex_double_to_complex_double, 12, work_size);
  //CHECK-NEXT:plan_mmany64_Z2Z->compute<sycl::double2, sycl::double2>(idata_mmany64_Z2Z, odata_mmany64_Z2Z, dpct::fft::fft_direction::backward);
  cufftHandle plan_mmany64_Z2Z;
  long long int odist_mmany64_Z2Z;
  long long int ostride_mmany64_Z2Z;
  long long int * onembed_mmany64_Z2Z;
  long long int idist_mmany64_Z2Z;
  long long int istride_mmany64_Z2Z;
  long long int* inembed_mmany64_Z2Z;
  long long int * n_mmany64_Z2Z;
  double2* odata_mmany64_Z2Z;
  double2* idata_mmany64_Z2Z;
  cufftMakePlanMany64(plan_mmany64_Z2Z, 3, n_mmany64_Z2Z, inembed_mmany64_Z2Z, istride_mmany64_Z2Z, idist_mmany64_Z2Z, onembed_mmany64_Z2Z, ostride_mmany64_Z2Z, odist_mmany64_Z2Z, CUFFT_Z2Z, 12, work_size);
  cufftExecZ2Z(plan_mmany64_Z2Z, idata_mmany64_Z2Z, odata_mmany64_Z2Z, CUFFT_INVERSE);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany64_Z2D;
  //CHECK-NEXT:long long int odist_mmany64_Z2D;
  //CHECK-NEXT:long long int ostride_mmany64_Z2D;
  //CHECK-NEXT:long long int * onembed_mmany64_Z2D;
  //CHECK-NEXT:long long int idist_mmany64_Z2D;
  //CHECK-NEXT:long long int istride_mmany64_Z2D;
  //CHECK-NEXT:long long int* inembed_mmany64_Z2D;
  //CHECK-NEXT:long long int * n_mmany64_Z2D;
  //CHECK-NEXT:double* odata_mmany64_Z2D;
  //CHECK-NEXT:sycl::double2* idata_mmany64_Z2D;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_Z2D->commit(&q_ct1, 3, n_mmany64_Z2D, inembed_mmany64_Z2D, istride_mmany64_Z2D, idist_mmany64_Z2D, onembed_mmany64_Z2D, ostride_mmany64_Z2D, odist_mmany64_Z2D, dpct::fft::fft_type::complex_double_to_real_double, 12, work_size);
  //CHECK-NEXT:plan_mmany64_Z2D->compute<sycl::double2, double>(idata_mmany64_Z2D, odata_mmany64_Z2D, dpct::fft::fft_direction::backward);
  cufftHandle plan_mmany64_Z2D;
  long long int odist_mmany64_Z2D;
  long long int ostride_mmany64_Z2D;
  long long int * onembed_mmany64_Z2D;
  long long int idist_mmany64_Z2D;
  long long int istride_mmany64_Z2D;
  long long int* inembed_mmany64_Z2D;
  long long int * n_mmany64_Z2D;
  double* odata_mmany64_Z2D;
  double2* idata_mmany64_Z2D;
  cufftMakePlanMany64(plan_mmany64_Z2D, 3, n_mmany64_Z2D, inembed_mmany64_Z2D, istride_mmany64_Z2D, idist_mmany64_Z2D, onembed_mmany64_Z2D, ostride_mmany64_Z2D, odist_mmany64_Z2D, CUFFT_Z2D, 12, work_size);
  cufftExecZ2D(plan_mmany64_Z2D, idata_mmany64_Z2D, odata_mmany64_Z2D);

  //CHECK:dpct::fft::fft_engine_ptr plan_mmany64_D2Z;
  //CHECK-NEXT:long long int odist_mmany64_D2Z;
  //CHECK-NEXT:long long int ostride_mmany64_D2Z;
  //CHECK-NEXT:long long int * onembed_mmany64_D2Z;
  //CHECK-NEXT:long long int idist_mmany64_D2Z;
  //CHECK-NEXT:long long int istride_mmany64_D2Z;
  //CHECK-NEXT:long long int* inembed_mmany64_D2Z;
  //CHECK-NEXT:long long int * n_mmany64_D2Z;
  //CHECK-NEXT:sycl::double2* odata_mmany64_D2Z;
  //CHECK-NEXT:double* idata_mmany64_D2Z;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported on GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_D2Z->commit(&q_ct1, 3, n_mmany64_D2Z, inembed_mmany64_D2Z, istride_mmany64_D2Z, idist_mmany64_D2Z, onembed_mmany64_D2Z, ostride_mmany64_D2Z, odist_mmany64_D2Z, dpct::fft::fft_type::real_double_to_complex_double, 12, work_size);
  //CHECK-NEXT:plan_mmany64_D2Z->compute<double, sycl::double2>(idata_mmany64_D2Z, odata_mmany64_D2Z, dpct::fft::fft_direction::forward);
  cufftHandle plan_mmany64_D2Z;
  long long int odist_mmany64_D2Z;
  long long int ostride_mmany64_D2Z;
  long long int * onembed_mmany64_D2Z;
  long long int idist_mmany64_D2Z;
  long long int istride_mmany64_D2Z;
  long long int* inembed_mmany64_D2Z;
  long long int * n_mmany64_D2Z;
  double2* odata_mmany64_D2Z;
  double* idata_mmany64_D2Z;
  cufftMakePlanMany64(plan_mmany64_D2Z, 3, n_mmany64_D2Z, inembed_mmany64_D2Z, istride_mmany64_D2Z, idist_mmany64_D2Z, onembed_mmany64_D2Z, ostride_mmany64_D2Z, odist_mmany64_D2Z, CUFFT_D2Z, 12, work_size);
  cufftExecD2Z(plan_mmany64_D2Z, idata_mmany64_D2Z, odata_mmany64_D2Z);

  return 0;
}

void foo() {
  int prop1, prop2, prop3;
  //CHECK:dpct::mkl_get_version(dpct::version_field::major, &prop1);
  //CHECK-NEXT:dpct::mkl_get_version(dpct::version_field::update, &prop2);
  //CHECK-NEXT:dpct::mkl_get_version(dpct::version_field::patch, &prop3);
  cufftGetProperty(MAJOR_VERSION, &prop1);
  cufftGetProperty(MINOR_VERSION, &prop2);
  cufftGetProperty(PATCH_LEVEL, &prop3);

  int ver;
  // CHECK: int err = CHECK_SYCL_ERROR(dpct::mkl_get_version(dpct::version_field::major, &ver));
  int err = cufftGetVersion(&ver);
}

