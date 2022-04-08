// RUN: c2s --format-range=none --usm-level=none -out-root %T/cufft %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft/cufft.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>

int main() {
  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>> plan_1d_C2C;
  //CHECK-NEXT:sycl::float2* odata_1d_C2C;
  //CHECK-NEXT:sycl::float2* idata_1d_C2C;
  cufftHandle plan_1d_C2C;
  float2* odata_1d_C2C;
  float2* idata_1d_C2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_C2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(10);
  //CHECK-NEXT:plan_1d_C2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_1d_C2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_1d_C2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10);
  //CHECK-NEXT:plan_1d_C2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftPlan1d(&plan_1d_C2C, 10, CUFFT_C2C, 3);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_C2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_1d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_1d_C2C);
  //CHECK-NEXT:if ((void *)idata_1d_C2C == (void *)odata_1d_C2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_1d_C2C, idata_1d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_1d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_1d_C2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_1d_C2C, idata_1d_C2C_buf_ct{{[0-9]+}}, odata_1d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2C(plan_1d_C2C, idata_1d_C2C, odata_1d_C2C, CUFFT_FORWARD);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_1d_C2R;
  //CHECK-NEXT:float* odata_1d_C2R;
  //CHECK-NEXT:sycl::float2* idata_1d_C2R;
  cufftHandle plan_1d_C2R;
  float* odata_1d_C2R;
  float2* idata_1d_C2R;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_C2R = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(10);
  //CHECK-NEXT:plan_1d_C2R->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:plan_1d_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_1d_C2R->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_1d_C2R->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10/2+1);
  //CHECK-NEXT:plan_1d_C2R->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftPlan1d(&plan_1d_C2R, 10, CUFFT_C2R, 3);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_C2R->commit(q_ct1);
  //CHECK-NEXT:auto idata_1d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_1d_C2R);
  //CHECK-NEXT:if ((void *)idata_1d_C2R == (void *)odata_1d_C2R) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_1d_C2R, idata_1d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_1d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_1d_C2R);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_1d_C2R, idata_1d_C2R_buf_ct{{[0-9]+}}, odata_1d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2R(plan_1d_C2R, idata_1d_C2R, odata_1d_C2R);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_1d_R2C;
  //CHECK-NEXT:sycl::float2* odata_1d_R2C;
  //CHECK-NEXT:float* idata_1d_R2C;
  cufftHandle plan_1d_R2C;
  float2* odata_1d_R2C;
  float* idata_1d_R2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_R2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(10);
  //CHECK-NEXT:plan_1d_R2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:plan_1d_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_1d_R2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_1d_R2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10/2+1);
  //CHECK-NEXT:plan_1d_R2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftPlan1d(&plan_1d_R2C, 10, CUFFT_R2C, 3);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_R2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_1d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_1d_R2C);
  //CHECK-NEXT:if ((void *)idata_1d_R2C == (void *)odata_1d_R2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_1d_R2C, idata_1d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_1d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_1d_R2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_1d_R2C, idata_1d_R2C_buf_ct{{[0-9]+}}, odata_1d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecR2C(plan_1d_R2C, idata_1d_R2C, odata_1d_R2C);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>> plan_1d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_1d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_1d_Z2Z;
  cufftHandle plan_1d_Z2Z;
  double2* odata_1d_Z2Z;
  double2* idata_1d_Z2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_Z2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>(10);
  //CHECK-NEXT:plan_1d_Z2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_1d_Z2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_1d_Z2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10);
  //CHECK-NEXT:plan_1d_Z2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftPlan1d(&plan_1d_Z2Z, 10, CUFFT_Z2Z, 3);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_Z2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_1d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_1d_Z2Z);
  //CHECK-NEXT:if ((void *)idata_1d_Z2Z == (void *)odata_1d_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_1d_Z2Z, idata_1d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_1d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_1d_Z2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_1d_Z2Z, idata_1d_Z2Z_buf_ct{{[0-9]+}}, odata_1d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_1d_Z2Z, idata_1d_Z2Z, odata_1d_Z2Z, CUFFT_INVERSE);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_1d_Z2D;
  //CHECK-NEXT:double* odata_1d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_1d_Z2D;
  cufftHandle plan_1d_Z2D;
  double* odata_1d_Z2D;
  double2* idata_1d_Z2D;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_Z2D = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
  //CHECK-NEXT:plan_1d_Z2D->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:plan_1d_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_1d_Z2D->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_1d_Z2D->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10/2+1);
  //CHECK-NEXT:plan_1d_Z2D->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftPlan1d(&plan_1d_Z2D, 10, CUFFT_Z2D, 3);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_Z2D->commit(q_ct1);
  //CHECK-NEXT:auto idata_1d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_1d_Z2D);
  //CHECK-NEXT:if ((void *)idata_1d_Z2D == (void *)odata_1d_Z2D) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_1d_Z2D, idata_1d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_1d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_1d_Z2D);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_1d_Z2D, idata_1d_Z2D_buf_ct{{[0-9]+}}, odata_1d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2D(plan_1d_Z2D, idata_1d_Z2D, odata_1d_Z2D);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_1d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_1d_D2Z;
  //CHECK-NEXT:double* idata_1d_D2Z;
  cufftHandle plan_1d_D2Z;
  double2* odata_1d_D2Z;
  double* idata_1d_D2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_D2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
  //CHECK-NEXT:plan_1d_D2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:plan_1d_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_1d_D2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_1d_D2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10/2+1);
  //CHECK-NEXT:plan_1d_D2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftPlan1d(&plan_1d_D2Z, 10, CUFFT_D2Z, 3);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_1d_D2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_1d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_1d_D2Z);
  //CHECK-NEXT:if ((void *)idata_1d_D2Z == (void *)odata_1d_D2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_1d_D2Z, idata_1d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_1d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_1d_D2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_1d_D2Z, idata_1d_D2Z_buf_ct{{[0-9]+}}, odata_1d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecD2Z(plan_1d_D2Z, idata_1d_D2Z, odata_1d_D2Z);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>> plan_2d_C2C;
  //CHECK-NEXT:sycl::float2* odata_2d_C2C;
  //CHECK-NEXT:sycl::float2* idata_2d_C2C;
  cufftHandle plan_2d_C2C;
  float2* odata_2d_C2C;
  float2* idata_2d_C2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_C2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_2d_C2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  cufftPlan2d(&plan_2d_C2C, 10, 20, CUFFT_C2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_C2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_2d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_2d_C2C);
  //CHECK-NEXT:if ((void *)idata_2d_C2C == (void *)odata_2d_C2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_2d_C2C, idata_2d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_2d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_2d_C2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_2d_C2C, idata_2d_C2C_buf_ct{{[0-9]+}}, odata_2d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2C(plan_2d_C2C, idata_2d_C2C, odata_2d_C2C, CUFFT_FORWARD);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_2d_C2R;
  //CHECK-NEXT:float* odata_2d_C2R;
  //CHECK-NEXT:sycl::float2* idata_2d_C2R;
  cufftHandle plan_2d_C2R;
  float* odata_2d_C2R;
  float2* idata_2d_C2R;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_C2R = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_2d_C2R->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[3] = {0, (20/2+1), 1};
  //CHECK-NEXT:plan_2d_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  cufftPlan2d(&plan_2d_C2R, 10, 20, CUFFT_C2R);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_C2R->commit(q_ct1);
  //CHECK-NEXT:auto idata_2d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_2d_C2R);
  //CHECK-NEXT:if ((void *)idata_2d_C2R == (void *)odata_2d_C2R) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_2d_C2R, idata_2d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_2d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_2d_C2R);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_2d_C2R, idata_2d_C2R_buf_ct{{[0-9]+}}, odata_2d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2R(plan_2d_C2R, idata_2d_C2R, odata_2d_C2R);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_2d_R2C;
  //CHECK-NEXT:sycl::float2* odata_2d_R2C;
  //CHECK-NEXT:float* idata_2d_R2C;
  cufftHandle plan_2d_R2C;
  float2* odata_2d_R2C;
  float* idata_2d_R2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_R2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_2d_R2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[3] = {0, (20/2+1), 1};
  //CHECK-NEXT:plan_2d_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  cufftPlan2d(&plan_2d_R2C, 10, 20, CUFFT_R2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_R2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_2d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_2d_R2C);
  //CHECK-NEXT:if ((void *)idata_2d_R2C == (void *)odata_2d_R2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_2d_R2C, idata_2d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_2d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_2d_R2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_2d_R2C, idata_2d_R2C_buf_ct{{[0-9]+}}, odata_2d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecR2C(plan_2d_R2C, idata_2d_R2C, odata_2d_R2C);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>> plan_2d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_2d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_2d_Z2Z;
  cufftHandle plan_2d_Z2Z;
  double2* odata_2d_Z2Z;
  double2* idata_2d_Z2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_Z2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_2d_Z2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  cufftPlan2d(&plan_2d_Z2Z, 10, 20, CUFFT_Z2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_Z2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_2d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_2d_Z2Z);
  //CHECK-NEXT:if ((void *)idata_2d_Z2Z == (void *)odata_2d_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_2d_Z2Z, idata_2d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_2d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_2d_Z2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_2d_Z2Z, idata_2d_Z2Z_buf_ct{{[0-9]+}}, odata_2d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_2d_Z2Z, idata_2d_Z2Z, odata_2d_Z2Z, CUFFT_INVERSE);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_2d_Z2D;
  //CHECK-NEXT:double* odata_2d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_2d_Z2D;
  cufftHandle plan_2d_Z2D;
  double* odata_2d_Z2D;
  double2* idata_2d_Z2D;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_Z2D = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_2d_Z2D->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[3] = {0, (20/2+1), 1};
  //CHECK-NEXT:plan_2d_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  cufftPlan2d(&plan_2d_Z2D, 10, 20, CUFFT_Z2D);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_Z2D->commit(q_ct1);
  //CHECK-NEXT:auto idata_2d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_2d_Z2D);
  //CHECK-NEXT:if ((void *)idata_2d_Z2D == (void *)odata_2d_Z2D) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_2d_Z2D, idata_2d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_2d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_2d_Z2D);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_2d_Z2D, idata_2d_Z2D_buf_ct{{[0-9]+}}, odata_2d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2D(plan_2d_Z2D, idata_2d_Z2D, odata_2d_Z2D);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_2d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_2d_D2Z;
  //CHECK-NEXT:double* idata_2d_D2Z;
  cufftHandle plan_2d_D2Z;
  double2* odata_2d_D2Z;
  double* idata_2d_D2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_D2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_2d_D2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[3] = {0, (20/2+1), 1};
  //CHECK-NEXT:plan_2d_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  cufftPlan2d(&plan_2d_D2Z, 10, 20, CUFFT_D2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_2d_D2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_2d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_2d_D2Z);
  //CHECK-NEXT:if ((void *)idata_2d_D2Z == (void *)odata_2d_D2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_2d_D2Z, idata_2d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_2d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_2d_D2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_2d_D2Z, idata_2d_D2Z_buf_ct{{[0-9]+}}, odata_2d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecD2Z(plan_2d_D2Z, idata_2d_D2Z, odata_2d_D2Z);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>> plan_3d_C2C;
  //CHECK-NEXT:sycl::float2* odata_3d_C2C;
  //CHECK-NEXT:sycl::float2* idata_3d_C2C;
  cufftHandle plan_3d_C2C;
  float2* odata_3d_C2C;
  float2* idata_3d_C2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_C2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_3d_C2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  cufftPlan3d(&plan_3d_C2C, 10, 20, 30, CUFFT_C2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_C2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_3d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_3d_C2C);
  //CHECK-NEXT:if ((void *)idata_3d_C2C == (void *)odata_3d_C2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_3d_C2C, idata_3d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_3d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_3d_C2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_3d_C2C, idata_3d_C2C_buf_ct{{[0-9]+}}, odata_3d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2C(plan_3d_C2C, idata_3d_C2C, odata_3d_C2C, CUFFT_FORWARD);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_3d_C2R;
  //CHECK-NEXT:float* odata_3d_C2R;
  //CHECK-NEXT:sycl::float2* idata_3d_C2R;
  cufftHandle plan_3d_C2R;
  float* odata_3d_C2R;
  float2* idata_3d_C2R;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_C2R = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_3d_C2R->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, 20*(30/2+1), (30/2+1), 1};
  //CHECK-NEXT:plan_3d_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  cufftPlan3d(&plan_3d_C2R, 10, 20, 30, CUFFT_C2R);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_C2R->commit(q_ct1);
  //CHECK-NEXT:auto idata_3d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_3d_C2R);
  //CHECK-NEXT:if ((void *)idata_3d_C2R == (void *)odata_3d_C2R) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_3d_C2R, idata_3d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_3d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_3d_C2R);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_3d_C2R, idata_3d_C2R_buf_ct{{[0-9]+}}, odata_3d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2R(plan_3d_C2R, idata_3d_C2R, odata_3d_C2R);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_3d_R2C;
  //CHECK-NEXT:sycl::float2* odata_3d_R2C;
  //CHECK-NEXT:float* idata_3d_R2C;
  cufftHandle plan_3d_R2C;
  float2* odata_3d_R2C;
  float* idata_3d_R2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_R2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_3d_R2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, 20*(30/2+1), (30/2+1), 1};
  //CHECK-NEXT:plan_3d_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  cufftPlan3d(&plan_3d_R2C, 10, 20, 30, CUFFT_R2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_R2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_3d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_3d_R2C);
  //CHECK-NEXT:if ((void *)idata_3d_R2C == (void *)odata_3d_R2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_3d_R2C, idata_3d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_3d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_3d_R2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_3d_R2C, idata_3d_R2C_buf_ct{{[0-9]+}}, odata_3d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecR2C(plan_3d_R2C, idata_3d_R2C, odata_3d_R2C);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>> plan_3d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_3d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_3d_Z2Z;
  cufftHandle plan_3d_Z2Z;
  double2* odata_3d_Z2Z;
  double2* idata_3d_Z2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_Z2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_3d_Z2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  cufftPlan3d(&plan_3d_Z2Z, 10, 20, 30, CUFFT_Z2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_Z2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_3d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_3d_Z2Z);
  //CHECK-NEXT:if ((void *)idata_3d_Z2Z == (void *)odata_3d_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_3d_Z2Z, idata_3d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_3d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_3d_Z2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_3d_Z2Z, idata_3d_Z2Z_buf_ct{{[0-9]+}}, odata_3d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_3d_Z2Z, idata_3d_Z2Z, odata_3d_Z2Z, CUFFT_INVERSE);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_3d_Z2D;
  //CHECK-NEXT:double* odata_3d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_3d_Z2D;
  cufftHandle plan_3d_Z2D;
  double* odata_3d_Z2D;
  double2* idata_3d_Z2D;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_Z2D = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_3d_Z2D->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, 20*(30/2+1), (30/2+1), 1};
  //CHECK-NEXT:plan_3d_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  cufftPlan3d(&plan_3d_Z2D, 10, 20, 30, CUFFT_Z2D);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_Z2D->commit(q_ct1);
  //CHECK-NEXT:auto idata_3d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_3d_Z2D);
  //CHECK-NEXT:if ((void *)idata_3d_Z2D == (void *)odata_3d_Z2D) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_3d_Z2D, idata_3d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_3d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_3d_Z2D);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_3d_Z2D, idata_3d_Z2D_buf_ct{{[0-9]+}}, odata_3d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2D(plan_3d_Z2D, idata_3d_Z2D, odata_3d_Z2D);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_3d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_3d_D2Z;
  //CHECK-NEXT:double* idata_3d_D2Z;
  cufftHandle plan_3d_D2Z;
  double2* odata_3d_D2Z;
  double* idata_3d_D2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_D2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_3d_D2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, 20*(30/2+1), (30/2+1), 1};
  //CHECK-NEXT:plan_3d_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  cufftPlan3d(&plan_3d_D2Z, 10, 20, 30, CUFFT_D2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_3d_D2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_3d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_3d_D2Z);
  //CHECK-NEXT:if ((void *)idata_3d_D2Z == (void *)odata_3d_D2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_3d_D2Z, idata_3d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_3d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_3d_D2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_3d_D2Z, idata_3d_D2Z_buf_ct{{[0-9]+}}, odata_3d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecD2Z(plan_3d_D2Z, idata_3d_D2Z, odata_3d_D2Z);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>> plan_many_C2C;
  //CHECK-NEXT:int odist_many_C2C;
  //CHECK-NEXT:int ostride_many_C2C;
  //CHECK-NEXT:int * onembed_many_C2C;
  //CHECK-NEXT:int idist_many_C2C;
  //CHECK-NEXT:int istride_many_C2C;
  //CHECK-NEXT:int* inembed_many_C2C;
  //CHECK-NEXT:int * n_many_C2C;
  //CHECK-NEXT:sycl::float2* odata_many_C2C;
  //CHECK-NEXT:sycl::float2* idata_many_C2C;
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

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_C2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{n_many_C2C[0], n_many_C2C[1], n_many_C2C[2]});
  //CHECK-NEXT:plan_many_C2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_many_C2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_many_C2C != nullptr && onembed_many_C2C != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_many_C2C[2] * inembed_many_C2C[1] * istride_many_C2C, inembed_many_C2C[2] * istride_many_C2C, istride_many_C2C};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_many_C2C[2] * onembed_many_C2C[1] * ostride_many_C2C, onembed_many_C2C[2] * ostride_many_C2C, ostride_many_C2C};
  //CHECK-NEXT:plan_many_C2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist_many_C2C);
  //CHECK-NEXT:plan_many_C2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist_many_C2C);
  //CHECK-NEXT:plan_many_C2C->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_C2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan_many_C2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_many_C2C[2]*n_many_C2C[1]*n_many_C2C[0]);
  //CHECK-NEXT:plan_many_C2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_many_C2C[2]*n_many_C2C[1]*n_many_C2C[0]);
  //CHECK-NEXT:}
  cufftPlanMany(&plan_many_C2C, 3, n_many_C2C, inembed_many_C2C, istride_many_C2C, idist_many_C2C, onembed_many_C2C, ostride_many_C2C, odist_many_C2C, CUFFT_C2C, 12);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_C2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_many_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_many_C2C);
  //CHECK-NEXT:if ((void *)idata_many_C2C == (void *)odata_many_C2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_many_C2C, idata_many_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_many_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_many_C2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_many_C2C, idata_many_C2C_buf_ct{{[0-9]+}}, odata_many_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2C(plan_many_C2C, idata_many_C2C, odata_many_C2C, CUFFT_FORWARD);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_many_C2R;
  //CHECK-NEXT:int odist_many_C2R;
  //CHECK-NEXT:int ostride_many_C2R;
  //CHECK-NEXT:int * onembed_many_C2R;
  //CHECK-NEXT:int idist_many_C2R;
  //CHECK-NEXT:int istride_many_C2R;
  //CHECK-NEXT:int* inembed_many_C2R;
  //CHECK-NEXT:int * n_many_C2R;
  //CHECK-NEXT:float* odata_many_C2R;
  //CHECK-NEXT:sycl::float2* idata_many_C2R;
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

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_C2R = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_many_C2R[0], n_many_C2R[1], n_many_C2R[2]});
  //CHECK-NEXT:plan_many_C2R->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_many_C2R->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_many_C2R != nullptr && onembed_many_C2R != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_many_C2R[2] * inembed_many_C2R[1] * istride_many_C2R, inembed_many_C2R[2] * istride_many_C2R, istride_many_C2R};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_many_C2R[2] * onembed_many_C2R[1] * ostride_many_C2R, onembed_many_C2R[2] * ostride_many_C2R, ostride_many_C2R};
  //CHECK-NEXT:plan_many_C2R->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist_many_C2R);
  //CHECK-NEXT:plan_many_C2R->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist_many_C2R);
  //CHECK-NEXT:plan_many_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_C2R->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n_many_C2R[1]*(n_many_C2R[2]/2+1), (n_many_C2R[2]/2+1), 1};
  //CHECK-NEXT:plan_many_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_C2R->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_many_C2R[2]*n_many_C2R[1]*n_many_C2R[0]);
  //CHECK-NEXT:plan_many_C2R->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_many_C2R[2]*n_many_C2R[1]*(n_many_C2R[0]/2+1));
  //CHECK-NEXT:}
  cufftPlanMany(&plan_many_C2R, 3, n_many_C2R, inembed_many_C2R, istride_many_C2R, idist_many_C2R, onembed_many_C2R, ostride_many_C2R, odist_many_C2R, CUFFT_C2R, 12);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_C2R->commit(q_ct1);
  //CHECK-NEXT:auto idata_many_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_many_C2R);
  //CHECK-NEXT:if ((void *)idata_many_C2R == (void *)odata_many_C2R) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_many_C2R, idata_many_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_many_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_many_C2R);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_many_C2R, idata_many_C2R_buf_ct{{[0-9]+}}, odata_many_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2R(plan_many_C2R, idata_many_C2R, odata_many_C2R);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_many_R2C;
  //CHECK-NEXT:int odist_many_R2C;
  //CHECK-NEXT:int ostride_many_R2C;
  //CHECK-NEXT:int * onembed_many_R2C;
  //CHECK-NEXT:int idist_many_R2C;
  //CHECK-NEXT:int istride_many_R2C;
  //CHECK-NEXT:int* inembed_many_R2C;
  //CHECK-NEXT:int * n_many_R2C;
  //CHECK-NEXT:sycl::float2* odata_many_R2C;
  //CHECK-NEXT:float* idata_many_R2C;
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

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_R2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_many_R2C[0], n_many_R2C[1], n_many_R2C[2]});
  //CHECK-NEXT:plan_many_R2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_many_R2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_many_R2C != nullptr && onembed_many_R2C != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_many_R2C[2] * inembed_many_R2C[1] * istride_many_R2C, inembed_many_R2C[2] * istride_many_R2C, istride_many_R2C};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_many_R2C[2] * onembed_many_R2C[1] * ostride_many_R2C, onembed_many_R2C[2] * ostride_many_R2C, ostride_many_R2C};
  //CHECK-NEXT:plan_many_R2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist_many_R2C);
  //CHECK-NEXT:plan_many_R2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist_many_R2C);
  //CHECK-NEXT:plan_many_R2C->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, n_many_R2C[1]*(n_many_R2C[2]/2+1), (n_many_R2C[2]/2+1), 1};
  //CHECK-NEXT:plan_many_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_R2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_many_R2C[2]*n_many_R2C[1]*n_many_R2C[0]);
  //CHECK-NEXT:plan_many_R2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_many_R2C[2]*n_many_R2C[1]*(n_many_R2C[0]/2+1));
  //CHECK-NEXT:}
  cufftPlanMany(&plan_many_R2C, 3, n_many_R2C, inembed_many_R2C, istride_many_R2C, idist_many_R2C, onembed_many_R2C, ostride_many_R2C, odist_many_R2C, CUFFT_R2C, 12);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_R2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_many_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_many_R2C);
  //CHECK-NEXT:if ((void *)idata_many_R2C == (void *)odata_many_R2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_many_R2C, idata_many_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_many_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_many_R2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_many_R2C, idata_many_R2C_buf_ct{{[0-9]+}}, odata_many_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecR2C(plan_many_R2C, idata_many_R2C, odata_many_R2C);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>> plan_many_Z2Z;
  //CHECK-NEXT:int odist_many_Z2Z;
  //CHECK-NEXT:int ostride_many_Z2Z;
  //CHECK-NEXT:int * onembed_many_Z2Z;
  //CHECK-NEXT:int idist_many_Z2Z;
  //CHECK-NEXT:int istride_many_Z2Z;
  //CHECK-NEXT:int* inembed_many_Z2Z;
  //CHECK-NEXT:int * n_many_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_many_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_many_Z2Z;
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

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_Z2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{n_many_Z2Z[0], n_many_Z2Z[1], n_many_Z2Z[2]});
  //CHECK-NEXT:plan_many_Z2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_many_Z2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_many_Z2Z != nullptr && onembed_many_Z2Z != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_many_Z2Z[2] * inembed_many_Z2Z[1] * istride_many_Z2Z, inembed_many_Z2Z[2] * istride_many_Z2Z, istride_many_Z2Z};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_many_Z2Z[2] * onembed_many_Z2Z[1] * ostride_many_Z2Z, onembed_many_Z2Z[2] * ostride_many_Z2Z, ostride_many_Z2Z};
  //CHECK-NEXT:plan_many_Z2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist_many_Z2Z);
  //CHECK-NEXT:plan_many_Z2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist_many_Z2Z);
  //CHECK-NEXT:plan_many_Z2Z->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_Z2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan_many_Z2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_many_Z2Z[2]*n_many_Z2Z[1]*n_many_Z2Z[0]);
  //CHECK-NEXT:plan_many_Z2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_many_Z2Z[2]*n_many_Z2Z[1]*n_many_Z2Z[0]);
  //CHECK-NEXT:}
  cufftPlanMany(&plan_many_Z2Z, 3, n_many_Z2Z, inembed_many_Z2Z, istride_many_Z2Z, idist_many_Z2Z, onembed_many_Z2Z, ostride_many_Z2Z, odist_many_Z2Z, CUFFT_Z2Z, 12);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_Z2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_many_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_many_Z2Z);
  //CHECK-NEXT:if ((void *)idata_many_Z2Z == (void *)odata_many_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_many_Z2Z, idata_many_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_many_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_many_Z2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_many_Z2Z, idata_many_Z2Z_buf_ct{{[0-9]+}}, odata_many_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_many_Z2Z, idata_many_Z2Z, odata_many_Z2Z, CUFFT_INVERSE);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_many_Z2D;
  //CHECK-NEXT:int odist_many_Z2D;
  //CHECK-NEXT:int ostride_many_Z2D;
  //CHECK-NEXT:int * onembed_many_Z2D;
  //CHECK-NEXT:int idist_many_Z2D;
  //CHECK-NEXT:int istride_many_Z2D;
  //CHECK-NEXT:int* inembed_many_Z2D;
  //CHECK-NEXT:int * n_many_Z2D;
  //CHECK-NEXT:double* odata_many_Z2D;
  //CHECK-NEXT:sycl::double2* idata_many_Z2D;
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

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_Z2D = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_many_Z2D[0], n_many_Z2D[1], n_many_Z2D[2]});
  //CHECK-NEXT:plan_many_Z2D->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_many_Z2D->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_many_Z2D != nullptr && onembed_many_Z2D != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_many_Z2D[2] * inembed_many_Z2D[1] * istride_many_Z2D, inembed_many_Z2D[2] * istride_many_Z2D, istride_many_Z2D};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_many_Z2D[2] * onembed_many_Z2D[1] * ostride_many_Z2D, onembed_many_Z2D[2] * ostride_many_Z2D, ostride_many_Z2D};
  //CHECK-NEXT:plan_many_Z2D->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist_many_Z2D);
  //CHECK-NEXT:plan_many_Z2D->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist_many_Z2D);
  //CHECK-NEXT:plan_many_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_Z2D->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n_many_Z2D[1]*(n_many_Z2D[2]/2+1), (n_many_Z2D[2]/2+1), 1};
  //CHECK-NEXT:plan_many_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_Z2D->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_many_Z2D[2]*n_many_Z2D[1]*n_many_Z2D[0]);
  //CHECK-NEXT:plan_many_Z2D->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_many_Z2D[2]*n_many_Z2D[1]*(n_many_Z2D[0]/2+1));
  //CHECK-NEXT:}
  cufftPlanMany(&plan_many_Z2D, 3, n_many_Z2D, inembed_many_Z2D, istride_many_Z2D, idist_many_Z2D, onembed_many_Z2D, ostride_many_Z2D, odist_many_Z2D, CUFFT_Z2D, 12);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_Z2D->commit(q_ct1);
  //CHECK-NEXT:auto idata_many_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_many_Z2D);
  //CHECK-NEXT:if ((void *)idata_many_Z2D == (void *)odata_many_Z2D) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_many_Z2D, idata_many_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_many_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_many_Z2D);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_many_Z2D, idata_many_Z2D_buf_ct{{[0-9]+}}, odata_many_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2D(plan_many_Z2D, idata_many_Z2D, odata_many_Z2D);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_many_D2Z;
  //CHECK-NEXT:int odist_many_D2Z;
  //CHECK-NEXT:int ostride_many_D2Z;
  //CHECK-NEXT:int * onembed_many_D2Z;
  //CHECK-NEXT:int idist_many_D2Z;
  //CHECK-NEXT:int istride_many_D2Z;
  //CHECK-NEXT:int* inembed_many_D2Z;
  //CHECK-NEXT:int * n_many_D2Z;
  //CHECK-NEXT:sycl::double2* odata_many_D2Z;
  //CHECK-NEXT:double* idata_many_D2Z;
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

  //CHECK:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_D2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_many_D2Z[0], n_many_D2Z[1], n_many_D2Z[2]});
  //CHECK-NEXT:plan_many_D2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_many_D2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_many_D2Z != nullptr && onembed_many_D2Z != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_many_D2Z[2] * inembed_many_D2Z[1] * istride_many_D2Z, inembed_many_D2Z[2] * istride_many_D2Z, istride_many_D2Z};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_many_D2Z[2] * onembed_many_D2Z[1] * ostride_many_D2Z, onembed_many_D2Z[2] * ostride_many_D2Z, ostride_many_D2Z};
  //CHECK-NEXT:plan_many_D2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist_many_D2Z);
  //CHECK-NEXT:plan_many_D2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist_many_D2Z);
  //CHECK-NEXT:plan_many_D2Z->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, n_many_D2Z[1]*(n_many_D2Z[2]/2+1), (n_many_D2Z[2]/2+1), 1};
  //CHECK-NEXT:plan_many_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_many_D2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_many_D2Z[2]*n_many_D2Z[1]*n_many_D2Z[0]);
  //CHECK-NEXT:plan_many_D2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_many_D2Z[2]*n_many_D2Z[1]*(n_many_D2Z[0]/2+1));
  //CHECK-NEXT:}
  cufftPlanMany(&plan_many_D2Z, 3, n_many_D2Z, inembed_many_D2Z, istride_many_D2Z, idist_many_D2Z, onembed_many_D2Z, ostride_many_D2Z, odist_many_D2Z, CUFFT_D2Z, 12);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_many_D2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_many_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_many_D2Z);
  //CHECK-NEXT:if ((void *)idata_many_D2Z == (void *)odata_many_D2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_many_D2Z, idata_many_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_many_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_many_D2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_many_D2Z, idata_many_D2Z_buf_ct{{[0-9]+}}, odata_many_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecD2Z(plan_many_D2Z, idata_many_D2Z, odata_many_D2Z);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>> plan_m1d_C2C;
  //CHECK-NEXT:size_t* work_size_m1d_C2C;
  //CHECK-NEXT:sycl::float2* odata_m1d_C2C;
  //CHECK-NEXT:sycl::float2* idata_m1d_C2C;
  cufftHandle plan_m1d_C2C;
  size_t* work_size_m1d_C2C;
  float2* odata_m1d_C2C;
  float2* idata_m1d_C2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m1d_C2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_C2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(10);
  //CHECK-NEXT:plan_m1d_C2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_m1d_C2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_m1d_C2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10);
  //CHECK-NEXT:plan_m1d_C2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftMakePlan1d(plan_m1d_C2C, 10, CUFFT_C2C, 3, work_size_m1d_C2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_C2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_m1d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_m1d_C2C);
  //CHECK-NEXT:if ((void *)idata_m1d_C2C == (void *)odata_m1d_C2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m1d_C2C, idata_m1d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m1d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_m1d_C2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m1d_C2C, idata_m1d_C2C_buf_ct{{[0-9]+}}, odata_m1d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2C(plan_m1d_C2C, idata_m1d_C2C, odata_m1d_C2C, CUFFT_FORWARD);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_m1d_C2R;
  //CHECK-NEXT:size_t* work_size_m1d_C2R;
  //CHECK-NEXT:float* odata_m1d_C2R;
  //CHECK-NEXT:sycl::float2* idata_m1d_C2R;
  cufftHandle plan_m1d_C2R;
  size_t* work_size_m1d_C2R;
  float* odata_m1d_C2R;
  float2* idata_m1d_C2R;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m1d_C2R' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_C2R = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(10);
  //CHECK-NEXT:plan_m1d_C2R->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:plan_m1d_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_m1d_C2R->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_m1d_C2R->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10/2+1);
  //CHECK-NEXT:plan_m1d_C2R->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftMakePlan1d(plan_m1d_C2R, 10, CUFFT_C2R, 3, work_size_m1d_C2R);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_C2R->commit(q_ct1);
  //CHECK-NEXT:auto idata_m1d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_m1d_C2R);
  //CHECK-NEXT:if ((void *)idata_m1d_C2R == (void *)odata_m1d_C2R) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m1d_C2R, idata_m1d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m1d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_m1d_C2R);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m1d_C2R, idata_m1d_C2R_buf_ct{{[0-9]+}}, odata_m1d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2R(plan_m1d_C2R, idata_m1d_C2R, odata_m1d_C2R);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_m1d_R2C;
  //CHECK-NEXT:size_t* work_size_m1d_R2C;
  //CHECK-NEXT:sycl::float2* odata_m1d_R2C;
  //CHECK-NEXT:float* idata_m1d_R2C;
  cufftHandle plan_m1d_R2C;
  size_t* work_size_m1d_R2C;
  float2* odata_m1d_R2C;
  float* idata_m1d_R2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m1d_R2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_R2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(10);
  //CHECK-NEXT:plan_m1d_R2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:plan_m1d_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_m1d_R2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_m1d_R2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10/2+1);
  //CHECK-NEXT:plan_m1d_R2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftMakePlan1d(plan_m1d_R2C, 10, CUFFT_R2C, 3, work_size_m1d_R2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_R2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_m1d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_m1d_R2C);
  //CHECK-NEXT:if ((void *)idata_m1d_R2C == (void *)odata_m1d_R2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m1d_R2C, idata_m1d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m1d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_m1d_R2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m1d_R2C, idata_m1d_R2C_buf_ct{{[0-9]+}}, odata_m1d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecR2C(plan_m1d_R2C, idata_m1d_R2C, odata_m1d_R2C);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>> plan_m1d_Z2Z;
  //CHECK-NEXT:size_t* work_size_m1d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_m1d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_m1d_Z2Z;
  cufftHandle plan_m1d_Z2Z;
  size_t* work_size_m1d_Z2Z;
  double2* odata_m1d_Z2Z;
  double2* idata_m1d_Z2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m1d_Z2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_Z2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>(10);
  //CHECK-NEXT:plan_m1d_Z2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_m1d_Z2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_m1d_Z2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10);
  //CHECK-NEXT:plan_m1d_Z2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftMakePlan1d(plan_m1d_Z2Z, 10, CUFFT_Z2Z, 3, work_size_m1d_Z2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_Z2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_m1d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_m1d_Z2Z);
  //CHECK-NEXT:if ((void *)idata_m1d_Z2Z == (void *)odata_m1d_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m1d_Z2Z, idata_m1d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m1d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_m1d_Z2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m1d_Z2Z, idata_m1d_Z2Z_buf_ct{{[0-9]+}}, odata_m1d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_m1d_Z2Z, idata_m1d_Z2Z, odata_m1d_Z2Z, CUFFT_INVERSE);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_m1d_Z2D;
  //CHECK-NEXT:size_t* work_size_m1d_Z2D;
  //CHECK-NEXT:double* odata_m1d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_m1d_Z2D;
  cufftHandle plan_m1d_Z2D;
  size_t* work_size_m1d_Z2D;
  double* odata_m1d_Z2D;
  double2* idata_m1d_Z2D;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m1d_Z2D' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_Z2D = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
  //CHECK-NEXT:plan_m1d_Z2D->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:plan_m1d_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_m1d_Z2D->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_m1d_Z2D->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10/2+1);
  //CHECK-NEXT:plan_m1d_Z2D->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftMakePlan1d(plan_m1d_Z2D, 10, CUFFT_Z2D, 3, work_size_m1d_Z2D);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_Z2D->commit(q_ct1);
  //CHECK-NEXT:auto idata_m1d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_m1d_Z2D);
  //CHECK-NEXT:if ((void *)idata_m1d_Z2D == (void *)odata_m1d_Z2D) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m1d_Z2D, idata_m1d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m1d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_m1d_Z2D);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m1d_Z2D, idata_m1d_Z2D_buf_ct{{[0-9]+}}, odata_m1d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2D(plan_m1d_Z2D, idata_m1d_Z2D, odata_m1d_Z2D);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_m1d_D2Z;
  //CHECK-NEXT:size_t* work_size_m1d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_m1d_D2Z;
  //CHECK-NEXT:double* idata_m1d_D2Z;
  cufftHandle plan_m1d_D2Z;
  size_t* work_size_m1d_D2Z;
  double2* odata_m1d_D2Z;
  double* idata_m1d_D2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m1d_D2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_D2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(10);
  //CHECK-NEXT:plan_m1d_D2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:plan_m1d_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_m1d_D2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, 10);
  //CHECK-NEXT:plan_m1d_D2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, 10/2+1);
  //CHECK-NEXT:plan_m1d_D2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  cufftMakePlan1d(plan_m1d_D2Z, 10, CUFFT_D2Z, 3, work_size_m1d_D2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m1d_D2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_m1d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_m1d_D2Z);
  //CHECK-NEXT:if ((void *)idata_m1d_D2Z == (void *)odata_m1d_D2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m1d_D2Z, idata_m1d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m1d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_m1d_D2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m1d_D2Z, idata_m1d_D2Z_buf_ct{{[0-9]+}}, odata_m1d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecD2Z(plan_m1d_D2Z, idata_m1d_D2Z, odata_m1d_D2Z);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>> plan_m2d_C2C;
  //CHECK-NEXT:size_t* work_size_m2d_C2C;
  //CHECK-NEXT:sycl::float2* odata_m2d_C2C;
  //CHECK-NEXT:sycl::float2* idata_m2d_C2C;
  cufftHandle plan_m2d_C2C;
  size_t* work_size_m2d_C2C;
  float2* odata_m2d_C2C;
  float2* idata_m2d_C2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m2d_C2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_C2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_m2d_C2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  cufftMakePlan2d(plan_m2d_C2C, 10, 20, CUFFT_C2C, work_size_m2d_C2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_C2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_m2d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_m2d_C2C);
  //CHECK-NEXT:if ((void *)idata_m2d_C2C == (void *)odata_m2d_C2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m2d_C2C, idata_m2d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m2d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_m2d_C2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m2d_C2C, idata_m2d_C2C_buf_ct{{[0-9]+}}, odata_m2d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2C(plan_m2d_C2C, idata_m2d_C2C, odata_m2d_C2C, CUFFT_FORWARD);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_m2d_C2R;
  //CHECK-NEXT:size_t* work_size_m2d_C2R;
  //CHECK-NEXT:float* odata_m2d_C2R;
  //CHECK-NEXT:sycl::float2* idata_m2d_C2R;
  cufftHandle plan_m2d_C2R;
  size_t* work_size_m2d_C2R;
  float* odata_m2d_C2R;
  float2* idata_m2d_C2R;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m2d_C2R' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_C2R = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_m2d_C2R->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[3] = {0, (20/2+1), 1};
  //CHECK-NEXT:plan_m2d_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  cufftMakePlan2d(plan_m2d_C2R, 10, 20, CUFFT_C2R, work_size_m2d_C2R);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_C2R->commit(q_ct1);
  //CHECK-NEXT:auto idata_m2d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_m2d_C2R);
  //CHECK-NEXT:if ((void *)idata_m2d_C2R == (void *)odata_m2d_C2R) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m2d_C2R, idata_m2d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m2d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_m2d_C2R);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m2d_C2R, idata_m2d_C2R_buf_ct{{[0-9]+}}, odata_m2d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2R(plan_m2d_C2R, idata_m2d_C2R, odata_m2d_C2R);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_m2d_R2C;
  //CHECK-NEXT:size_t* work_size_m2d_R2C;
  //CHECK-NEXT:sycl::float2* odata_m2d_R2C;
  //CHECK-NEXT:float* idata_m2d_R2C;
  cufftHandle plan_m2d_R2C;
  size_t* work_size_m2d_R2C;
  float2* odata_m2d_R2C;
  float* idata_m2d_R2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m2d_R2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_R2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_m2d_R2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[3] = {0, (20/2+1), 1};
  //CHECK-NEXT:plan_m2d_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  cufftMakePlan2d(plan_m2d_R2C, 10, 20, CUFFT_R2C, work_size_m2d_R2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_R2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_m2d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_m2d_R2C);
  //CHECK-NEXT:if ((void *)idata_m2d_R2C == (void *)odata_m2d_R2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m2d_R2C, idata_m2d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m2d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_m2d_R2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m2d_R2C, idata_m2d_R2C_buf_ct{{[0-9]+}}, odata_m2d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecR2C(plan_m2d_R2C, idata_m2d_R2C, odata_m2d_R2C);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>> plan_m2d_Z2Z;
  //CHECK-NEXT:size_t* work_size_m2d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_m2d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_m2d_Z2Z;
  cufftHandle plan_m2d_Z2Z;
  size_t* work_size_m2d_Z2Z;
  double2* odata_m2d_Z2Z;
  double2* idata_m2d_Z2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m2d_Z2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_Z2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_m2d_Z2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  cufftMakePlan2d(plan_m2d_Z2Z, 10, 20, CUFFT_Z2Z, work_size_m2d_Z2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_Z2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_m2d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_m2d_Z2Z);
  //CHECK-NEXT:if ((void *)idata_m2d_Z2Z == (void *)odata_m2d_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m2d_Z2Z, idata_m2d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m2d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_m2d_Z2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m2d_Z2Z, idata_m2d_Z2Z_buf_ct{{[0-9]+}}, odata_m2d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_m2d_Z2Z, idata_m2d_Z2Z, odata_m2d_Z2Z, CUFFT_INVERSE);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_m2d_Z2D;
  //CHECK-NEXT:size_t* work_size_m2d_Z2D;
  //CHECK-NEXT:double* odata_m2d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_m2d_Z2D;
  cufftHandle plan_m2d_Z2D;
  size_t* work_size_m2d_Z2D;
  double* odata_m2d_Z2D;
  double2* idata_m2d_Z2D;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m2d_Z2D' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_Z2D = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_m2d_Z2D->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[3] = {0, (20/2+1), 1};
  //CHECK-NEXT:plan_m2d_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  cufftMakePlan2d(plan_m2d_Z2D, 10, 20, CUFFT_Z2D, work_size_m2d_Z2D);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_Z2D->commit(q_ct1);
  //CHECK-NEXT:auto idata_m2d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_m2d_Z2D);
  //CHECK-NEXT:if ((void *)idata_m2d_Z2D == (void *)odata_m2d_Z2D) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m2d_Z2D, idata_m2d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m2d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_m2d_Z2D);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m2d_Z2D, idata_m2d_Z2D_buf_ct{{[0-9]+}}, odata_m2d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2D(plan_m2d_Z2D, idata_m2d_Z2D, odata_m2d_Z2D);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_m2d_D2Z;
  //CHECK-NEXT:size_t* work_size_m2d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_m2d_D2Z;
  //CHECK-NEXT:double* idata_m2d_D2Z;
  cufftHandle plan_m2d_D2Z;
  size_t* work_size_m2d_D2Z;
  double2* odata_m2d_D2Z;
  double* idata_m2d_D2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m2d_D2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_D2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20});
  //CHECK-NEXT:plan_m2d_D2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[3] = {0, (20/2+1), 1};
  //CHECK-NEXT:plan_m2d_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  cufftMakePlan2d(plan_m2d_D2Z, 10, 20, CUFFT_D2Z, work_size_m2d_D2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m2d_D2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_m2d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_m2d_D2Z);
  //CHECK-NEXT:if ((void *)idata_m2d_D2Z == (void *)odata_m2d_D2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m2d_D2Z, idata_m2d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m2d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_m2d_D2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m2d_D2Z, idata_m2d_D2Z_buf_ct{{[0-9]+}}, odata_m2d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecD2Z(plan_m2d_D2Z, idata_m2d_D2Z, odata_m2d_D2Z);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>> plan_m3d_C2C;
  //CHECK-NEXT:size_t* work_size_m3d_C2C;
  //CHECK-NEXT:sycl::float2* odata_m3d_C2C;
  //CHECK-NEXT:sycl::float2* idata_m3d_C2C;
  cufftHandle plan_m3d_C2C;
  size_t* work_size_m3d_C2C;
  float2* odata_m3d_C2C;
  float2* idata_m3d_C2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m3d_C2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_C2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_m3d_C2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  cufftMakePlan3d(plan_m3d_C2C, 10, 20, 30, CUFFT_C2C, work_size_m3d_C2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_C2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_m3d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_m3d_C2C);
  //CHECK-NEXT:if ((void *)idata_m3d_C2C == (void *)odata_m3d_C2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m3d_C2C, idata_m3d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m3d_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_m3d_C2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m3d_C2C, idata_m3d_C2C_buf_ct{{[0-9]+}}, odata_m3d_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2C(plan_m3d_C2C, idata_m3d_C2C, odata_m3d_C2C, CUFFT_FORWARD);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_m3d_C2R;
  //CHECK-NEXT:size_t* work_size_m3d_C2R;
  //CHECK-NEXT:float* odata_m3d_C2R;
  //CHECK-NEXT:sycl::float2* idata_m3d_C2R;
  cufftHandle plan_m3d_C2R;
  size_t* work_size_m3d_C2R;
  float* odata_m3d_C2R;
  float2* idata_m3d_C2R;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m3d_C2R' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_C2R = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_m3d_C2R->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, 20*(30/2+1), (30/2+1), 1};
  //CHECK-NEXT:plan_m3d_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  cufftMakePlan3d(plan_m3d_C2R, 10, 20, 30, CUFFT_C2R, work_size_m3d_C2R);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_C2R->commit(q_ct1);
  //CHECK-NEXT:auto idata_m3d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_m3d_C2R);
  //CHECK-NEXT:if ((void *)idata_m3d_C2R == (void *)odata_m3d_C2R) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m3d_C2R, idata_m3d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m3d_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_m3d_C2R);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m3d_C2R, idata_m3d_C2R_buf_ct{{[0-9]+}}, odata_m3d_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2R(plan_m3d_C2R, idata_m3d_C2R, odata_m3d_C2R);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_m3d_R2C;
  //CHECK-NEXT:size_t* work_size_m3d_R2C;
  //CHECK-NEXT:sycl::float2* odata_m3d_R2C;
  //CHECK-NEXT:float* idata_m3d_R2C;
  cufftHandle plan_m3d_R2C;
  size_t* work_size_m3d_R2C;
  float2* odata_m3d_R2C;
  float* idata_m3d_R2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m3d_R2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_R2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_m3d_R2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, 20*(30/2+1), (30/2+1), 1};
  //CHECK-NEXT:plan_m3d_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  cufftMakePlan3d(plan_m3d_R2C, 10, 20, 30, CUFFT_R2C, work_size_m3d_R2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_R2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_m3d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_m3d_R2C);
  //CHECK-NEXT:if ((void *)idata_m3d_R2C == (void *)odata_m3d_R2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m3d_R2C, idata_m3d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m3d_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_m3d_R2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m3d_R2C, idata_m3d_R2C_buf_ct{{[0-9]+}}, odata_m3d_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecR2C(plan_m3d_R2C, idata_m3d_R2C, odata_m3d_R2C);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>> plan_m3d_Z2Z;
  //CHECK-NEXT:size_t* work_size_m3d_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_m3d_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_m3d_Z2Z;
  cufftHandle plan_m3d_Z2Z;
  size_t* work_size_m3d_Z2Z;
  double2* odata_m3d_Z2Z;
  double2* idata_m3d_Z2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m3d_Z2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_Z2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_m3d_Z2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  cufftMakePlan3d(plan_m3d_Z2Z, 10, 20, 30, CUFFT_Z2Z, work_size_m3d_Z2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_Z2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_m3d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_m3d_Z2Z);
  //CHECK-NEXT:if ((void *)idata_m3d_Z2Z == (void *)odata_m3d_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m3d_Z2Z, idata_m3d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m3d_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_m3d_Z2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m3d_Z2Z, idata_m3d_Z2Z_buf_ct{{[0-9]+}}, odata_m3d_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_m3d_Z2Z, idata_m3d_Z2Z, odata_m3d_Z2Z, CUFFT_INVERSE);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_m3d_Z2D;
  //CHECK-NEXT:size_t* work_size_m3d_Z2D;
  //CHECK-NEXT:double* odata_m3d_Z2D;
  //CHECK-NEXT:sycl::double2* idata_m3d_Z2D;
  cufftHandle plan_m3d_Z2D;
  size_t* work_size_m3d_Z2D;
  double* odata_m3d_Z2D;
  double2* idata_m3d_Z2D;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m3d_Z2D' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_Z2D = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_m3d_Z2D->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, 20*(30/2+1), (30/2+1), 1};
  //CHECK-NEXT:plan_m3d_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  cufftMakePlan3d(plan_m3d_Z2D, 10, 20, 30, CUFFT_Z2D, work_size_m3d_Z2D);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_Z2D->commit(q_ct1);
  //CHECK-NEXT:auto idata_m3d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_m3d_Z2D);
  //CHECK-NEXT:if ((void *)idata_m3d_Z2D == (void *)odata_m3d_Z2D) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m3d_Z2D, idata_m3d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m3d_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_m3d_Z2D);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_m3d_Z2D, idata_m3d_Z2D_buf_ct{{[0-9]+}}, odata_m3d_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2D(plan_m3d_Z2D, idata_m3d_Z2D, odata_m3d_Z2D);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_m3d_D2Z;
  //CHECK-NEXT:size_t* work_size_m3d_D2Z;
  //CHECK-NEXT:sycl::double2* odata_m3d_D2Z;
  //CHECK-NEXT:double* idata_m3d_D2Z;
  cufftHandle plan_m3d_D2Z;
  size_t* work_size_m3d_D2Z;
  double2* odata_m3d_D2Z;
  double* idata_m3d_D2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_m3d_D2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_D2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{10, 20, 30});
  //CHECK-NEXT:plan_m3d_D2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, 20*(30/2+1), (30/2+1), 1};
  //CHECK-NEXT:plan_m3d_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  cufftMakePlan3d(plan_m3d_D2Z, 10, 20, 30, CUFFT_D2Z, work_size_m3d_D2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_m3d_D2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_m3d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_m3d_D2Z);
  //CHECK-NEXT:if ((void *)idata_m3d_D2Z == (void *)odata_m3d_D2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m3d_D2Z, idata_m3d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_m3d_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_m3d_D2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_m3d_D2Z, idata_m3d_D2Z_buf_ct{{[0-9]+}}, odata_m3d_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecD2Z(plan_m3d_D2Z, idata_m3d_D2Z, odata_m3d_D2Z);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>> plan_mmany_C2C;
  //CHECK-NEXT:size_t* work_size_mmany_C2C;
  //CHECK-NEXT:int odist_mmany_C2C;
  //CHECK-NEXT:int ostride_mmany_C2C;
  //CHECK-NEXT:int * onembed_mmany_C2C;
  //CHECK-NEXT:int idist_mmany_C2C;
  //CHECK-NEXT:int istride_mmany_C2C;
  //CHECK-NEXT:int* inembed_mmany_C2C;
  //CHECK-NEXT:int * n_mmany_C2C;
  //CHECK-NEXT:sycl::float2* odata_mmany_C2C;
  //CHECK-NEXT:sycl::float2* idata_mmany_C2C;
  cufftHandle plan_mmany_C2C;
  size_t* work_size_mmany_C2C;
  int odist_mmany_C2C;
  int ostride_mmany_C2C;
  int * onembed_mmany_C2C;
  int idist_mmany_C2C;
  int istride_mmany_C2C;
  int* inembed_mmany_C2C;
  int * n_mmany_C2C;
  float2* odata_mmany_C2C;
  float2* idata_mmany_C2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany_C2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_C2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{n_mmany_C2C[0], n_mmany_C2C[1], n_mmany_C2C[2]});
  //CHECK-NEXT:plan_mmany_C2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany_C2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany_C2C != nullptr && onembed_mmany_C2C != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany_C2C[2] * inembed_mmany_C2C[1] * istride_mmany_C2C, inembed_mmany_C2C[2] * istride_mmany_C2C, istride_mmany_C2C};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany_C2C[2] * onembed_mmany_C2C[1] * ostride_mmany_C2C, onembed_mmany_C2C[2] * ostride_mmany_C2C, ostride_mmany_C2C};
  //CHECK-NEXT:plan_mmany_C2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist_mmany_C2C);
  //CHECK-NEXT:plan_mmany_C2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist_mmany_C2C);
  //CHECK-NEXT:plan_mmany_C2C->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_C2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan_mmany_C2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany_C2C[2]*n_mmany_C2C[1]*n_mmany_C2C[0]);
  //CHECK-NEXT:plan_mmany_C2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany_C2C[2]*n_mmany_C2C[1]*n_mmany_C2C[0]);
  //CHECK-NEXT:}
  cufftMakePlanMany(plan_mmany_C2C, 3, n_mmany_C2C, inembed_mmany_C2C, istride_mmany_C2C, idist_mmany_C2C, onembed_mmany_C2C, ostride_mmany_C2C, odist_mmany_C2C, CUFFT_C2C, 12, work_size_mmany_C2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_C2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_mmany_C2C);
  //CHECK-NEXT:if ((void *)idata_mmany_C2C == (void *)odata_mmany_C2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany_C2C, idata_mmany_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_mmany_C2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany_C2C, idata_mmany_C2C_buf_ct{{[0-9]+}}, odata_mmany_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2C(plan_mmany_C2C, idata_mmany_C2C, odata_mmany_C2C, CUFFT_FORWARD);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_mmany_C2R;
  //CHECK-NEXT:size_t* work_size_mmany_C2R;
  //CHECK-NEXT:int odist_mmany_C2R;
  //CHECK-NEXT:int ostride_mmany_C2R;
  //CHECK-NEXT:int * onembed_mmany_C2R;
  //CHECK-NEXT:int idist_mmany_C2R;
  //CHECK-NEXT:int istride_mmany_C2R;
  //CHECK-NEXT:int* inembed_mmany_C2R;
  //CHECK-NEXT:int * n_mmany_C2R;
  //CHECK-NEXT:float* odata_mmany_C2R;
  //CHECK-NEXT:sycl::float2* idata_mmany_C2R;
  cufftHandle plan_mmany_C2R;
  size_t* work_size_mmany_C2R;
  int odist_mmany_C2R;
  int ostride_mmany_C2R;
  int * onembed_mmany_C2R;
  int idist_mmany_C2R;
  int istride_mmany_C2R;
  int* inembed_mmany_C2R;
  int * n_mmany_C2R;
  float* odata_mmany_C2R;
  float2* idata_mmany_C2R;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany_C2R' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_C2R = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_mmany_C2R[0], n_mmany_C2R[1], n_mmany_C2R[2]});
  //CHECK-NEXT:plan_mmany_C2R->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany_C2R->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany_C2R != nullptr && onembed_mmany_C2R != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany_C2R[2] * inembed_mmany_C2R[1] * istride_mmany_C2R, inembed_mmany_C2R[2] * istride_mmany_C2R, istride_mmany_C2R};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany_C2R[2] * onembed_mmany_C2R[1] * ostride_mmany_C2R, onembed_mmany_C2R[2] * ostride_mmany_C2R, ostride_mmany_C2R};
  //CHECK-NEXT:plan_mmany_C2R->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist_mmany_C2R);
  //CHECK-NEXT:plan_mmany_C2R->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist_mmany_C2R);
  //CHECK-NEXT:plan_mmany_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_C2R->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n_mmany_C2R[1]*(n_mmany_C2R[2]/2+1), (n_mmany_C2R[2]/2+1), 1};
  //CHECK-NEXT:plan_mmany_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_C2R->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany_C2R[2]*n_mmany_C2R[1]*n_mmany_C2R[0]);
  //CHECK-NEXT:plan_mmany_C2R->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany_C2R[2]*n_mmany_C2R[1]*(n_mmany_C2R[0]/2+1));
  //CHECK-NEXT:}
  cufftMakePlanMany(plan_mmany_C2R, 3, n_mmany_C2R, inembed_mmany_C2R, istride_mmany_C2R, idist_mmany_C2R, onembed_mmany_C2R, ostride_mmany_C2R, odist_mmany_C2R, CUFFT_C2R, 12, work_size_mmany_C2R);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_C2R->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_mmany_C2R);
  //CHECK-NEXT:if ((void *)idata_mmany_C2R == (void *)odata_mmany_C2R) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany_C2R, idata_mmany_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_mmany_C2R);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany_C2R, idata_mmany_C2R_buf_ct{{[0-9]+}}, odata_mmany_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2R(plan_mmany_C2R, idata_mmany_C2R, odata_mmany_C2R);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_mmany_R2C;
  //CHECK-NEXT:size_t* work_size_mmany_R2C;
  //CHECK-NEXT:int odist_mmany_R2C;
  //CHECK-NEXT:int ostride_mmany_R2C;
  //CHECK-NEXT:int * onembed_mmany_R2C;
  //CHECK-NEXT:int idist_mmany_R2C;
  //CHECK-NEXT:int istride_mmany_R2C;
  //CHECK-NEXT:int* inembed_mmany_R2C;
  //CHECK-NEXT:int * n_mmany_R2C;
  //CHECK-NEXT:sycl::float2* odata_mmany_R2C;
  //CHECK-NEXT:float* idata_mmany_R2C;
  cufftHandle plan_mmany_R2C;
  size_t* work_size_mmany_R2C;
  int odist_mmany_R2C;
  int ostride_mmany_R2C;
  int * onembed_mmany_R2C;
  int idist_mmany_R2C;
  int istride_mmany_R2C;
  int* inembed_mmany_R2C;
  int * n_mmany_R2C;
  float2* odata_mmany_R2C;
  float* idata_mmany_R2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany_R2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_R2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_mmany_R2C[0], n_mmany_R2C[1], n_mmany_R2C[2]});
  //CHECK-NEXT:plan_mmany_R2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany_R2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany_R2C != nullptr && onembed_mmany_R2C != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany_R2C[2] * inembed_mmany_R2C[1] * istride_mmany_R2C, inembed_mmany_R2C[2] * istride_mmany_R2C, istride_mmany_R2C};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany_R2C[2] * onembed_mmany_R2C[1] * ostride_mmany_R2C, onembed_mmany_R2C[2] * ostride_mmany_R2C, ostride_mmany_R2C};
  //CHECK-NEXT:plan_mmany_R2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist_mmany_R2C);
  //CHECK-NEXT:plan_mmany_R2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist_mmany_R2C);
  //CHECK-NEXT:plan_mmany_R2C->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, n_mmany_R2C[1]*(n_mmany_R2C[2]/2+1), (n_mmany_R2C[2]/2+1), 1};
  //CHECK-NEXT:plan_mmany_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_R2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany_R2C[2]*n_mmany_R2C[1]*n_mmany_R2C[0]);
  //CHECK-NEXT:plan_mmany_R2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany_R2C[2]*n_mmany_R2C[1]*(n_mmany_R2C[0]/2+1));
  //CHECK-NEXT:}
  cufftMakePlanMany(plan_mmany_R2C, 3, n_mmany_R2C, inembed_mmany_R2C, istride_mmany_R2C, idist_mmany_R2C, onembed_mmany_R2C, ostride_mmany_R2C, odist_mmany_R2C, CUFFT_R2C, 12, work_size_mmany_R2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_R2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_mmany_R2C);
  //CHECK-NEXT:if ((void *)idata_mmany_R2C == (void *)odata_mmany_R2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany_R2C, idata_mmany_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_mmany_R2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany_R2C, idata_mmany_R2C_buf_ct{{[0-9]+}}, odata_mmany_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecR2C(plan_mmany_R2C, idata_mmany_R2C, odata_mmany_R2C);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>> plan_mmany_Z2Z;
  //CHECK-NEXT:size_t* work_size_mmany_Z2Z;
  //CHECK-NEXT:int odist_mmany_Z2Z;
  //CHECK-NEXT:int ostride_mmany_Z2Z;
  //CHECK-NEXT:int * onembed_mmany_Z2Z;
  //CHECK-NEXT:int idist_mmany_Z2Z;
  //CHECK-NEXT:int istride_mmany_Z2Z;
  //CHECK-NEXT:int* inembed_mmany_Z2Z;
  //CHECK-NEXT:int * n_mmany_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_mmany_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_mmany_Z2Z;
  cufftHandle plan_mmany_Z2Z;
  size_t* work_size_mmany_Z2Z;
  int odist_mmany_Z2Z;
  int ostride_mmany_Z2Z;
  int * onembed_mmany_Z2Z;
  int idist_mmany_Z2Z;
  int istride_mmany_Z2Z;
  int* inembed_mmany_Z2Z;
  int * n_mmany_Z2Z;
  double2* odata_mmany_Z2Z;
  double2* idata_mmany_Z2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany_Z2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_Z2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{n_mmany_Z2Z[0], n_mmany_Z2Z[1], n_mmany_Z2Z[2]});
  //CHECK-NEXT:plan_mmany_Z2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany_Z2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany_Z2Z != nullptr && onembed_mmany_Z2Z != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany_Z2Z[2] * inembed_mmany_Z2Z[1] * istride_mmany_Z2Z, inembed_mmany_Z2Z[2] * istride_mmany_Z2Z, istride_mmany_Z2Z};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany_Z2Z[2] * onembed_mmany_Z2Z[1] * ostride_mmany_Z2Z, onembed_mmany_Z2Z[2] * ostride_mmany_Z2Z, ostride_mmany_Z2Z};
  //CHECK-NEXT:plan_mmany_Z2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist_mmany_Z2Z);
  //CHECK-NEXT:plan_mmany_Z2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist_mmany_Z2Z);
  //CHECK-NEXT:plan_mmany_Z2Z->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_Z2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan_mmany_Z2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany_Z2Z[2]*n_mmany_Z2Z[1]*n_mmany_Z2Z[0]);
  //CHECK-NEXT:plan_mmany_Z2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany_Z2Z[2]*n_mmany_Z2Z[1]*n_mmany_Z2Z[0]);
  //CHECK-NEXT:}
  cufftMakePlanMany(plan_mmany_Z2Z, 3, n_mmany_Z2Z, inembed_mmany_Z2Z, istride_mmany_Z2Z, idist_mmany_Z2Z, onembed_mmany_Z2Z, ostride_mmany_Z2Z, odist_mmany_Z2Z, CUFFT_Z2Z, 12, work_size_mmany_Z2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_Z2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_mmany_Z2Z);
  //CHECK-NEXT:if ((void *)idata_mmany_Z2Z == (void *)odata_mmany_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany_Z2Z, idata_mmany_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_mmany_Z2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany_Z2Z, idata_mmany_Z2Z_buf_ct{{[0-9]+}}, odata_mmany_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_mmany_Z2Z, idata_mmany_Z2Z, odata_mmany_Z2Z, CUFFT_INVERSE);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_mmany_Z2D;
  //CHECK-NEXT:size_t* work_size_mmany_Z2D;
  //CHECK-NEXT:int odist_mmany_Z2D;
  //CHECK-NEXT:int ostride_mmany_Z2D;
  //CHECK-NEXT:int * onembed_mmany_Z2D;
  //CHECK-NEXT:int idist_mmany_Z2D;
  //CHECK-NEXT:int istride_mmany_Z2D;
  //CHECK-NEXT:int* inembed_mmany_Z2D;
  //CHECK-NEXT:int * n_mmany_Z2D;
  //CHECK-NEXT:double* odata_mmany_Z2D;
  //CHECK-NEXT:sycl::double2* idata_mmany_Z2D;
  cufftHandle plan_mmany_Z2D;
  size_t* work_size_mmany_Z2D;
  int odist_mmany_Z2D;
  int ostride_mmany_Z2D;
  int * onembed_mmany_Z2D;
  int idist_mmany_Z2D;
  int istride_mmany_Z2D;
  int* inembed_mmany_Z2D;
  int * n_mmany_Z2D;
  double* odata_mmany_Z2D;
  double2* idata_mmany_Z2D;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany_Z2D' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_Z2D = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_mmany_Z2D[0], n_mmany_Z2D[1], n_mmany_Z2D[2]});
  //CHECK-NEXT:plan_mmany_Z2D->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany_Z2D->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany_Z2D != nullptr && onembed_mmany_Z2D != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany_Z2D[2] * inembed_mmany_Z2D[1] * istride_mmany_Z2D, inembed_mmany_Z2D[2] * istride_mmany_Z2D, istride_mmany_Z2D};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany_Z2D[2] * onembed_mmany_Z2D[1] * ostride_mmany_Z2D, onembed_mmany_Z2D[2] * ostride_mmany_Z2D, ostride_mmany_Z2D};
  //CHECK-NEXT:plan_mmany_Z2D->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist_mmany_Z2D);
  //CHECK-NEXT:plan_mmany_Z2D->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist_mmany_Z2D);
  //CHECK-NEXT:plan_mmany_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_Z2D->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n_mmany_Z2D[1]*(n_mmany_Z2D[2]/2+1), (n_mmany_Z2D[2]/2+1), 1};
  //CHECK-NEXT:plan_mmany_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_Z2D->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany_Z2D[2]*n_mmany_Z2D[1]*n_mmany_Z2D[0]);
  //CHECK-NEXT:plan_mmany_Z2D->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany_Z2D[2]*n_mmany_Z2D[1]*(n_mmany_Z2D[0]/2+1));
  //CHECK-NEXT:}
  cufftMakePlanMany(plan_mmany_Z2D, 3, n_mmany_Z2D, inembed_mmany_Z2D, istride_mmany_Z2D, idist_mmany_Z2D, onembed_mmany_Z2D, ostride_mmany_Z2D, odist_mmany_Z2D, CUFFT_Z2D, 12, work_size_mmany_Z2D);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_Z2D->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_mmany_Z2D);
  //CHECK-NEXT:if ((void *)idata_mmany_Z2D == (void *)odata_mmany_Z2D) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany_Z2D, idata_mmany_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_mmany_Z2D);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany_Z2D, idata_mmany_Z2D_buf_ct{{[0-9]+}}, odata_mmany_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2D(plan_mmany_Z2D, idata_mmany_Z2D, odata_mmany_Z2D);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_mmany_D2Z;
  //CHECK-NEXT:size_t* work_size_mmany_D2Z;
  //CHECK-NEXT:int odist_mmany_D2Z;
  //CHECK-NEXT:int ostride_mmany_D2Z;
  //CHECK-NEXT:int * onembed_mmany_D2Z;
  //CHECK-NEXT:int idist_mmany_D2Z;
  //CHECK-NEXT:int istride_mmany_D2Z;
  //CHECK-NEXT:int* inembed_mmany_D2Z;
  //CHECK-NEXT:int * n_mmany_D2Z;
  //CHECK-NEXT:sycl::double2* odata_mmany_D2Z;
  //CHECK-NEXT:double* idata_mmany_D2Z;
  cufftHandle plan_mmany_D2Z;
  size_t* work_size_mmany_D2Z;
  int odist_mmany_D2Z;
  int ostride_mmany_D2Z;
  int * onembed_mmany_D2Z;
  int idist_mmany_D2Z;
  int istride_mmany_D2Z;
  int* inembed_mmany_D2Z;
  int * n_mmany_D2Z;
  double2* odata_mmany_D2Z;
  double* idata_mmany_D2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany_D2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_D2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_mmany_D2Z[0], n_mmany_D2Z[1], n_mmany_D2Z[2]});
  //CHECK-NEXT:plan_mmany_D2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany_D2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany_D2Z != nullptr && onembed_mmany_D2Z != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany_D2Z[2] * inembed_mmany_D2Z[1] * istride_mmany_D2Z, inembed_mmany_D2Z[2] * istride_mmany_D2Z, istride_mmany_D2Z};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany_D2Z[2] * onembed_mmany_D2Z[1] * ostride_mmany_D2Z, onembed_mmany_D2Z[2] * ostride_mmany_D2Z, ostride_mmany_D2Z};
  //CHECK-NEXT:plan_mmany_D2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist_mmany_D2Z);
  //CHECK-NEXT:plan_mmany_D2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist_mmany_D2Z);
  //CHECK-NEXT:plan_mmany_D2Z->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, n_mmany_D2Z[1]*(n_mmany_D2Z[2]/2+1), (n_mmany_D2Z[2]/2+1), 1};
  //CHECK-NEXT:plan_mmany_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany_D2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany_D2Z[2]*n_mmany_D2Z[1]*n_mmany_D2Z[0]);
  //CHECK-NEXT:plan_mmany_D2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany_D2Z[2]*n_mmany_D2Z[1]*(n_mmany_D2Z[0]/2+1));
  //CHECK-NEXT:}
  cufftMakePlanMany(plan_mmany_D2Z, 3, n_mmany_D2Z, inembed_mmany_D2Z, istride_mmany_D2Z, idist_mmany_D2Z, onembed_mmany_D2Z, ostride_mmany_D2Z, odist_mmany_D2Z, CUFFT_D2Z, 12, work_size_mmany_D2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany_D2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_mmany_D2Z);
  //CHECK-NEXT:if ((void *)idata_mmany_D2Z == (void *)odata_mmany_D2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany_D2Z, idata_mmany_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_mmany_D2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany_D2Z, idata_mmany_D2Z_buf_ct{{[0-9]+}}, odata_mmany_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecD2Z(plan_mmany_D2Z, idata_mmany_D2Z, odata_mmany_D2Z);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>> plan_mmany64_C2C;
  //CHECK-NEXT:size_t* work_size_mmany64_C2C;
  //CHECK-NEXT:long long int odist_mmany64_C2C;
  //CHECK-NEXT:long long int ostride_mmany64_C2C;
  //CHECK-NEXT:long long int * onembed_mmany64_C2C;
  //CHECK-NEXT:long long int idist_mmany64_C2C;
  //CHECK-NEXT:long long int istride_mmany64_C2C;
  //CHECK-NEXT:long long int* inembed_mmany64_C2C;
  //CHECK-NEXT:long long int * n_mmany64_C2C;
  //CHECK-NEXT:sycl::float2* odata_mmany64_C2C;
  //CHECK-NEXT:sycl::float2* idata_mmany64_C2C;
  cufftHandle plan_mmany64_C2C;
  size_t* work_size_mmany64_C2C;
  long long int odist_mmany64_C2C;
  long long int ostride_mmany64_C2C;
  long long int * onembed_mmany64_C2C;
  long long int idist_mmany64_C2C;
  long long int istride_mmany64_C2C;
  long long int* inembed_mmany64_C2C;
  long long int * n_mmany64_C2C;
  float2* odata_mmany64_C2C;
  float2* idata_mmany64_C2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany64_C2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_C2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{n_mmany64_C2C[0], n_mmany64_C2C[1], n_mmany64_C2C[2]});
  //CHECK-NEXT:plan_mmany64_C2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany64_C2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany64_C2C != nullptr && onembed_mmany64_C2C != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany64_C2C[2] * inembed_mmany64_C2C[1] * istride_mmany64_C2C, inembed_mmany64_C2C[2] * istride_mmany64_C2C, istride_mmany64_C2C};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany64_C2C[2] * onembed_mmany64_C2C[1] * ostride_mmany64_C2C, onembed_mmany64_C2C[2] * ostride_mmany64_C2C, ostride_mmany64_C2C};
  //CHECK-NEXT:plan_mmany64_C2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist_mmany64_C2C);
  //CHECK-NEXT:plan_mmany64_C2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist_mmany64_C2C);
  //CHECK-NEXT:plan_mmany64_C2C->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_C2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan_mmany64_C2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany64_C2C[2]*n_mmany64_C2C[1]*n_mmany64_C2C[0]);
  //CHECK-NEXT:plan_mmany64_C2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany64_C2C[2]*n_mmany64_C2C[1]*n_mmany64_C2C[0]);
  //CHECK-NEXT:}
  cufftMakePlanMany64(plan_mmany64_C2C, 3, n_mmany64_C2C, inembed_mmany64_C2C, istride_mmany64_C2C, idist_mmany64_C2C, onembed_mmany64_C2C, ostride_mmany64_C2C, odist_mmany64_C2C, CUFFT_C2C, 12, work_size_mmany64_C2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_C2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany64_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_mmany64_C2C);
  //CHECK-NEXT:if ((void *)idata_mmany64_C2C == (void *)odata_mmany64_C2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany64_C2C, idata_mmany64_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany64_C2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_mmany64_C2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany64_C2C, idata_mmany64_C2C_buf_ct{{[0-9]+}}, odata_mmany64_C2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2C(plan_mmany64_C2C, idata_mmany64_C2C, odata_mmany64_C2C, CUFFT_FORWARD);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_mmany64_C2R;
  //CHECK-NEXT:size_t* work_size_mmany64_C2R;
  //CHECK-NEXT:long long int odist_mmany64_C2R;
  //CHECK-NEXT:long long int ostride_mmany64_C2R;
  //CHECK-NEXT:long long int * onembed_mmany64_C2R;
  //CHECK-NEXT:long long int idist_mmany64_C2R;
  //CHECK-NEXT:long long int istride_mmany64_C2R;
  //CHECK-NEXT:long long int* inembed_mmany64_C2R;
  //CHECK-NEXT:long long int * n_mmany64_C2R;
  //CHECK-NEXT:float* odata_mmany64_C2R;
  //CHECK-NEXT:sycl::float2* idata_mmany64_C2R;
  cufftHandle plan_mmany64_C2R;
  size_t* work_size_mmany64_C2R;
  long long int odist_mmany64_C2R;
  long long int ostride_mmany64_C2R;
  long long int * onembed_mmany64_C2R;
  long long int idist_mmany64_C2R;
  long long int istride_mmany64_C2R;
  long long int* inembed_mmany64_C2R;
  long long int * n_mmany64_C2R;
  float* odata_mmany64_C2R;
  float2* idata_mmany64_C2R;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany64_C2R' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_C2R = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_mmany64_C2R[0], n_mmany64_C2R[1], n_mmany64_C2R[2]});
  //CHECK-NEXT:plan_mmany64_C2R->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany64_C2R->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany64_C2R != nullptr && onembed_mmany64_C2R != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany64_C2R[2] * inembed_mmany64_C2R[1] * istride_mmany64_C2R, inembed_mmany64_C2R[2] * istride_mmany64_C2R, istride_mmany64_C2R};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany64_C2R[2] * onembed_mmany64_C2R[1] * ostride_mmany64_C2R, onembed_mmany64_C2R[2] * ostride_mmany64_C2R, ostride_mmany64_C2R};
  //CHECK-NEXT:plan_mmany64_C2R->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist_mmany64_C2R);
  //CHECK-NEXT:plan_mmany64_C2R->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist_mmany64_C2R);
  //CHECK-NEXT:plan_mmany64_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_C2R->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n_mmany64_C2R[1]*(n_mmany64_C2R[2]/2+1), (n_mmany64_C2R[2]/2+1), 1};
  //CHECK-NEXT:plan_mmany64_C2R->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_C2R->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany64_C2R[2]*n_mmany64_C2R[1]*n_mmany64_C2R[0]);
  //CHECK-NEXT:plan_mmany64_C2R->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany64_C2R[2]*n_mmany64_C2R[1]*(n_mmany64_C2R[0]/2+1));
  //CHECK-NEXT:}
  cufftMakePlanMany64(plan_mmany64_C2R, 3, n_mmany64_C2R, inembed_mmany64_C2R, istride_mmany64_C2R, idist_mmany64_C2R, onembed_mmany64_C2R, ostride_mmany64_C2R, odist_mmany64_C2R, CUFFT_C2R, 12, work_size_mmany64_C2R);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_C2R->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany64_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_mmany64_C2R);
  //CHECK-NEXT:if ((void *)idata_mmany64_C2R == (void *)odata_mmany64_C2R) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany64_C2R, idata_mmany64_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany64_C2R_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_mmany64_C2R);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany64_C2R, idata_mmany64_C2R_buf_ct{{[0-9]+}}, odata_mmany64_C2R_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecC2R(plan_mmany64_C2R, idata_mmany64_C2R, odata_mmany64_C2R);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan_mmany64_R2C;
  //CHECK-NEXT:size_t* work_size_mmany64_R2C;
  //CHECK-NEXT:long long int odist_mmany64_R2C;
  //CHECK-NEXT:long long int ostride_mmany64_R2C;
  //CHECK-NEXT:long long int * onembed_mmany64_R2C;
  //CHECK-NEXT:long long int idist_mmany64_R2C;
  //CHECK-NEXT:long long int istride_mmany64_R2C;
  //CHECK-NEXT:long long int* inembed_mmany64_R2C;
  //CHECK-NEXT:long long int * n_mmany64_R2C;
  //CHECK-NEXT:sycl::float2* odata_mmany64_R2C;
  //CHECK-NEXT:float* idata_mmany64_R2C;
  cufftHandle plan_mmany64_R2C;
  size_t* work_size_mmany64_R2C;
  long long int odist_mmany64_R2C;
  long long int ostride_mmany64_R2C;
  long long int * onembed_mmany64_R2C;
  long long int idist_mmany64_R2C;
  long long int istride_mmany64_R2C;
  long long int* inembed_mmany64_R2C;
  long long int * n_mmany64_R2C;
  float2* odata_mmany64_R2C;
  float* idata_mmany64_R2C;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany64_R2C' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_R2C = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_mmany64_R2C[0], n_mmany64_R2C[1], n_mmany64_R2C[2]});
  //CHECK-NEXT:plan_mmany64_R2C->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany64_R2C->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany64_R2C != nullptr && onembed_mmany64_R2C != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany64_R2C[2] * inembed_mmany64_R2C[1] * istride_mmany64_R2C, inembed_mmany64_R2C[2] * istride_mmany64_R2C, istride_mmany64_R2C};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany64_R2C[2] * onembed_mmany64_R2C[1] * ostride_mmany64_R2C, onembed_mmany64_R2C[2] * ostride_mmany64_R2C, ostride_mmany64_R2C};
  //CHECK-NEXT:plan_mmany64_R2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist_mmany64_R2C);
  //CHECK-NEXT:plan_mmany64_R2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist_mmany64_R2C);
  //CHECK-NEXT:plan_mmany64_R2C->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, n_mmany64_R2C[1]*(n_mmany64_R2C[2]/2+1), (n_mmany64_R2C[2]/2+1), 1};
  //CHECK-NEXT:plan_mmany64_R2C->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_R2C->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany64_R2C[2]*n_mmany64_R2C[1]*n_mmany64_R2C[0]);
  //CHECK-NEXT:plan_mmany64_R2C->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany64_R2C[2]*n_mmany64_R2C[1]*(n_mmany64_R2C[0]/2+1));
  //CHECK-NEXT:}
  cufftMakePlanMany64(plan_mmany64_R2C, 3, n_mmany64_R2C, inembed_mmany64_R2C, istride_mmany64_R2C, idist_mmany64_R2C, onembed_mmany64_R2C, ostride_mmany64_R2C, odist_mmany64_R2C, CUFFT_R2C, 12, work_size_mmany64_R2C);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_R2C->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany64_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(idata_mmany64_R2C);
  //CHECK-NEXT:if ((void *)idata_mmany64_R2C == (void *)odata_mmany64_R2C) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany64_R2C, idata_mmany64_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany64_R2C_buf_ct{{[0-9]+}} = c2s::get_buffer<float>(odata_mmany64_R2C);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany64_R2C, idata_mmany64_R2C_buf_ct{{[0-9]+}}, odata_mmany64_R2C_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecR2C(plan_mmany64_R2C, idata_mmany64_R2C, odata_mmany64_R2C);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>> plan_mmany64_Z2Z;
  //CHECK-NEXT:size_t* work_size_mmany64_Z2Z;
  //CHECK-NEXT:long long int odist_mmany64_Z2Z;
  //CHECK-NEXT:long long int ostride_mmany64_Z2Z;
  //CHECK-NEXT:long long int * onembed_mmany64_Z2Z;
  //CHECK-NEXT:long long int idist_mmany64_Z2Z;
  //CHECK-NEXT:long long int istride_mmany64_Z2Z;
  //CHECK-NEXT:long long int* inembed_mmany64_Z2Z;
  //CHECK-NEXT:long long int * n_mmany64_Z2Z;
  //CHECK-NEXT:sycl::double2* odata_mmany64_Z2Z;
  //CHECK-NEXT:sycl::double2* idata_mmany64_Z2Z;
  cufftHandle plan_mmany64_Z2Z;
  size_t* work_size_mmany64_Z2Z;
  long long int odist_mmany64_Z2Z;
  long long int ostride_mmany64_Z2Z;
  long long int * onembed_mmany64_Z2Z;
  long long int idist_mmany64_Z2Z;
  long long int istride_mmany64_Z2Z;
  long long int* inembed_mmany64_Z2Z;
  long long int * n_mmany64_Z2Z;
  double2* odata_mmany64_Z2Z;
  double2* idata_mmany64_Z2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany64_Z2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_Z2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>(std::vector<std::int64_t>{n_mmany64_Z2Z[0], n_mmany64_Z2Z[1], n_mmany64_Z2Z[2]});
  //CHECK-NEXT:plan_mmany64_Z2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany64_Z2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany64_Z2Z != nullptr && onembed_mmany64_Z2Z != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany64_Z2Z[2] * inembed_mmany64_Z2Z[1] * istride_mmany64_Z2Z, inembed_mmany64_Z2Z[2] * istride_mmany64_Z2Z, istride_mmany64_Z2Z};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany64_Z2Z[2] * onembed_mmany64_Z2Z[1] * ostride_mmany64_Z2Z, onembed_mmany64_Z2Z[2] * ostride_mmany64_Z2Z, ostride_mmany64_Z2Z};
  //CHECK-NEXT:plan_mmany64_Z2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist_mmany64_Z2Z);
  //CHECK-NEXT:plan_mmany64_Z2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist_mmany64_Z2Z);
  //CHECK-NEXT:plan_mmany64_Z2Z->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_Z2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan_mmany64_Z2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany64_Z2Z[2]*n_mmany64_Z2Z[1]*n_mmany64_Z2Z[0]);
  //CHECK-NEXT:plan_mmany64_Z2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany64_Z2Z[2]*n_mmany64_Z2Z[1]*n_mmany64_Z2Z[0]);
  //CHECK-NEXT:}
  cufftMakePlanMany64(plan_mmany64_Z2Z, 3, n_mmany64_Z2Z, inembed_mmany64_Z2Z, istride_mmany64_Z2Z, idist_mmany64_Z2Z, onembed_mmany64_Z2Z, ostride_mmany64_Z2Z, odist_mmany64_Z2Z, CUFFT_Z2Z, 12, work_size_mmany64_Z2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_Z2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany64_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_mmany64_Z2Z);
  //CHECK-NEXT:if ((void *)idata_mmany64_Z2Z == (void *)odata_mmany64_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany64_Z2Z, idata_mmany64_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany64_Z2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_mmany64_Z2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany64_Z2Z, idata_mmany64_Z2Z_buf_ct{{[0-9]+}}, odata_mmany64_Z2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_mmany64_Z2Z, idata_mmany64_Z2Z, odata_mmany64_Z2Z, CUFFT_INVERSE);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_mmany64_Z2D;
  //CHECK-NEXT:size_t* work_size_mmany64_Z2D;
  //CHECK-NEXT:long long int odist_mmany64_Z2D;
  //CHECK-NEXT:long long int ostride_mmany64_Z2D;
  //CHECK-NEXT:long long int * onembed_mmany64_Z2D;
  //CHECK-NEXT:long long int idist_mmany64_Z2D;
  //CHECK-NEXT:long long int istride_mmany64_Z2D;
  //CHECK-NEXT:long long int* inembed_mmany64_Z2D;
  //CHECK-NEXT:long long int * n_mmany64_Z2D;
  //CHECK-NEXT:double* odata_mmany64_Z2D;
  //CHECK-NEXT:sycl::double2* idata_mmany64_Z2D;
  cufftHandle plan_mmany64_Z2D;
  size_t* work_size_mmany64_Z2D;
  long long int odist_mmany64_Z2D;
  long long int ostride_mmany64_Z2D;
  long long int * onembed_mmany64_Z2D;
  long long int idist_mmany64_Z2D;
  long long int istride_mmany64_Z2D;
  long long int* inembed_mmany64_Z2D;
  long long int * n_mmany64_Z2D;
  double* odata_mmany64_Z2D;
  double2* idata_mmany64_Z2D;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany64_Z2D' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_Z2D = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_mmany64_Z2D[0], n_mmany64_Z2D[1], n_mmany64_Z2D[2]});
  //CHECK-NEXT:plan_mmany64_Z2D->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany64_Z2D->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany64_Z2D != nullptr && onembed_mmany64_Z2D != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany64_Z2D[2] * inembed_mmany64_Z2D[1] * istride_mmany64_Z2D, inembed_mmany64_Z2D[2] * istride_mmany64_Z2D, istride_mmany64_Z2D};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany64_Z2D[2] * onembed_mmany64_Z2D[1] * ostride_mmany64_Z2D, onembed_mmany64_Z2D[2] * ostride_mmany64_Z2D, ostride_mmany64_Z2D};
  //CHECK-NEXT:plan_mmany64_Z2D->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist_mmany64_Z2D);
  //CHECK-NEXT:plan_mmany64_Z2D->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist_mmany64_Z2D);
  //CHECK-NEXT:plan_mmany64_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_Z2D->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n_mmany64_Z2D[1]*(n_mmany64_Z2D[2]/2+1), (n_mmany64_Z2D[2]/2+1), 1};
  //CHECK-NEXT:plan_mmany64_Z2D->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_Z2D->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany64_Z2D[2]*n_mmany64_Z2D[1]*n_mmany64_Z2D[0]);
  //CHECK-NEXT:plan_mmany64_Z2D->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany64_Z2D[2]*n_mmany64_Z2D[1]*(n_mmany64_Z2D[0]/2+1));
  //CHECK-NEXT:}
  cufftMakePlanMany64(plan_mmany64_Z2D, 3, n_mmany64_Z2D, inembed_mmany64_Z2D, istride_mmany64_Z2D, idist_mmany64_Z2D, onembed_mmany64_Z2D, ostride_mmany64_Z2D, odist_mmany64_Z2D, CUFFT_Z2D, 12, work_size_mmany64_Z2D);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_Z2D->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany64_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_mmany64_Z2D);
  //CHECK-NEXT:if ((void *)idata_mmany64_Z2D == (void *)odata_mmany64_Z2D) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany64_Z2D, idata_mmany64_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany64_Z2D_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_mmany64_Z2D);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany64_Z2D, idata_mmany64_Z2D_buf_ct{{[0-9]+}}, odata_mmany64_Z2D_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecZ2D(plan_mmany64_Z2D, idata_mmany64_Z2D, odata_mmany64_Z2D);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan_mmany64_D2Z;
  //CHECK-NEXT:size_t* work_size_mmany64_D2Z;
  //CHECK-NEXT:long long int odist_mmany64_D2Z;
  //CHECK-NEXT:long long int ostride_mmany64_D2Z;
  //CHECK-NEXT:long long int * onembed_mmany64_D2Z;
  //CHECK-NEXT:long long int idist_mmany64_D2Z;
  //CHECK-NEXT:long long int istride_mmany64_D2Z;
  //CHECK-NEXT:long long int* inembed_mmany64_D2Z;
  //CHECK-NEXT:long long int * n_mmany64_D2Z;
  //CHECK-NEXT:sycl::double2* odata_mmany64_D2Z;
  //CHECK-NEXT:double* idata_mmany64_D2Z;
  cufftHandle plan_mmany64_D2Z;
  size_t* work_size_mmany64_D2Z;
  long long int odist_mmany64_D2Z;
  long long int ostride_mmany64_D2Z;
  long long int * onembed_mmany64_D2Z;
  long long int idist_mmany64_D2Z;
  long long int istride_mmany64_D2Z;
  long long int* inembed_mmany64_D2Z;
  long long int * n_mmany64_D2Z;
  double2* odata_mmany64_D2Z;
  double* idata_mmany64_D2Z;

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size_mmany64_D2Z' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_D2Z = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n_mmany64_D2Z[0], n_mmany64_D2Z[1], n_mmany64_D2Z[2]});
  //CHECK-NEXT:plan_mmany64_D2Z->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan_mmany64_D2Z->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed_mmany64_D2Z != nullptr && onembed_mmany64_D2Z != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed_mmany64_D2Z[2] * inembed_mmany64_D2Z[1] * istride_mmany64_D2Z, inembed_mmany64_D2Z[2] * istride_mmany64_D2Z, istride_mmany64_D2Z};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed_mmany64_D2Z[2] * onembed_mmany64_D2Z[1] * ostride_mmany64_D2Z, onembed_mmany64_D2Z[2] * ostride_mmany64_D2Z, ostride_mmany64_D2Z};
  //CHECK-NEXT:plan_mmany64_D2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist_mmany64_D2Z);
  //CHECK-NEXT:plan_mmany64_D2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, odist_mmany64_D2Z);
  //CHECK-NEXT:plan_mmany64_D2Z->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, n_mmany64_D2Z[1]*(n_mmany64_D2Z[2]/2+1), (n_mmany64_D2Z[2]/2+1), 1};
  //CHECK-NEXT:plan_mmany64_D2Z->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan_mmany64_D2Z->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n_mmany64_D2Z[2]*n_mmany64_D2Z[1]*n_mmany64_D2Z[0]);
  //CHECK-NEXT:plan_mmany64_D2Z->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n_mmany64_D2Z[2]*n_mmany64_D2Z[1]*(n_mmany64_D2Z[0]/2+1));
  //CHECK-NEXT:}
  cufftMakePlanMany64(plan_mmany64_D2Z, 3, n_mmany64_D2Z, inembed_mmany64_D2Z, istride_mmany64_D2Z, idist_mmany64_D2Z, onembed_mmany64_D2Z, ostride_mmany64_D2Z, odist_mmany64_D2Z, CUFFT_D2Z, 12, work_size_mmany64_D2Z);

  //CHECK:{
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_D2Z->commit(q_ct1);
  //CHECK-NEXT:auto idata_mmany64_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(idata_mmany64_D2Z);
  //CHECK-NEXT:if ((void *)idata_mmany64_D2Z == (void *)odata_mmany64_D2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany64_D2Z, idata_mmany64_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:auto odata_mmany64_D2Z_buf_ct{{[0-9]+}} = c2s::get_buffer<double>(odata_mmany64_D2Z);
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan_mmany64_D2Z, idata_mmany64_D2Z_buf_ct{{[0-9]+}}, odata_mmany64_D2Z_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftExecD2Z(plan_mmany64_D2Z, idata_mmany64_D2Z, odata_mmany64_D2Z);

  return 0;
}