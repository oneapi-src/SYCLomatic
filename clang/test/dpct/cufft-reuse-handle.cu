// RUN: dpct --format-range=none -out-root %T/cufft-reuse-handle %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-reuse-handle/cufft-reuse-handle.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>

int main() {
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

  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9+]}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_Z2Z->commit(dpct::get_default_queue());
  //CHECK-NEXT:if ((void *)idata_mmany64_Z2Z == (void *)odata_mmany64_Z2Z) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany64_Z2Z, (double*)idata_mmany64_Z2Z);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan_mmany64_Z2Z, (double*)idata_mmany64_Z2Z, (double*)odata_mmany64_Z2Z);
  //CHECK-NEXT:}
  cufftExecZ2Z(plan_mmany64_Z2Z, idata_mmany64_Z2Z, odata_mmany64_Z2Z, CUFFT_INVERSE);

  return 0;
}
