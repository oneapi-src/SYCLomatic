// RUN: dpct --format-range=none -out-root %T/cufft-different-locations-usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-different-locations-usm/cufft-different-locations-usm.dp.cpp --match-full-lines %s
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
double* odata;
double2* idata;

#define HANDLE_CUFFT_ERROR( err ) (CufftHandleError( err, __FILE__, __LINE__ ))
static void CufftHandleError( cufftResult err, const char *file, int line ) {
  if (err != CUFFT_SUCCESS) {
    fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",
            __FILE__, __LINE__, "error" );
  }
}

int main() {
  cufftHandle plan1;
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int res1 = (plan1 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]}), 0);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  cufftResult res1 = cufftMakePlanMany(plan1, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  //CHECK:int res2 = 0;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan1->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan1, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:res2 = (oneapi::mkl::dft::compute_backward(*plan1, (double*)idata, odata), 0);
  //CHECK-NEXT:}
  cufftResult res2 = cufftExecZ2D(plan1, idata, odata);

  cufftHandle plan2;
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:res1 = (plan2 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]}), 0);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  res1 = cufftMakePlanMany(plan2, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan2->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan2, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:res2 = (oneapi::mkl::dft::compute_backward(*plan2, (double*)idata, odata), 0);
  //CHECK-NEXT:}
  res2 = cufftExecZ2D(plan2, idata, odata);

  cufftHandle plan3;
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan3 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in function-like macro statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:HANDLE_CUFFT_ERROR(0);
  HANDLE_CUFFT_ERROR(cufftMakePlanMany(plan3, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  //CHECK:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:HANDLE_CUFFT_ERROR([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan3->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan3, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan3, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  HANDLE_CUFFT_ERROR(cufftExecZ2D(plan3, idata, odata));

  cufftHandle plan4;
  cufftHandle plan5;
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan4 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan4->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan4->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan4->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan4->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan4->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan4->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan4->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan4->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan4->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(0) {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} else if ([&](){
  //CHECK-NEXT:plan5 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan5->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan5->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan5->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan5->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan5->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan5->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan5->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan5->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan5->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}()) {
  //CHECK-NEXT:}
  if(cufftMakePlanMany(plan4, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  } else if (cufftMakePlanMany(plan5, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan4->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan4, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan4, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if (0) {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} else if([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan5->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan5, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan5, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}()) {
  //CHECK-NEXT:}
  if (cufftExecZ2D(plan4, idata, odata)) {
  } else if(cufftExecZ2D(plan5, idata, odata)) {
  }

  cufftHandle plan6;
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan6 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan6->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan6->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan6->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan6->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan6->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan6->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan6->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan6->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan6->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(int res = 0) {
  //CHECK-NEXT:}
  if(cufftResult res = cufftMakePlanMany(plan6, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan6->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan6, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan6, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(int res = 0) {
  //CHECK-NEXT:}
  if(cufftResult res = cufftExecZ2D(plan6, idata, odata)) {
  }

  cufftHandle plan7;
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan7 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan7->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan7->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan7->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan7->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan7->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan7->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan7->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan7->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan7->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a for statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (0;;) {
  //CHECK-NEXT:}
  for (cufftMakePlanMany(plan7, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);;) {
  }
  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan7->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan7, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan7, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a for statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (0;;) {
  //CHECK-NEXT:}
  for (cufftExecZ2D(plan7, idata, odata);;) {
  }

  cufftHandle plan8;
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (;[&](){
  //CHECK-NEXT:plan8 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan8->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan8->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan8->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan8->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan8->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan8->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan8->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan8->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan8->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}();) {
  //CHECK-NEXT:}
  for (;cufftMakePlanMany(plan8, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);) {
  }

  //CHECK:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (;[&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan8->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan8, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan8, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}();) {
  //CHECK-NEXT:}
  for (;cufftExecZ2D(plan8, idata, odata);) {
  }

  cufftHandle plan9;
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:while ([&](){
  //CHECK-NEXT:plan9 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan9->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan9->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan9->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan9->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan9->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan9->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan9->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan9->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan9->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}() != 0) {
  //CHECK-NEXT:}
  while (cufftMakePlanMany(plan9, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size) != 0) {
  }

  //CHECK:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:while ([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan9->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan9, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan9, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}() != 0) {
  //CHECK-NEXT:}
  while (cufftExecZ2D(plan9, idata, odata) != 0) {
  }

  cufftHandle plan10;
  //CHECK:do {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} while ([&](){
  //CHECK-NEXT:plan10 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan10->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan10->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan10->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan10->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan10->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan10->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan10->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan10->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan10->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  do {
  } while (cufftMakePlanMany(plan10, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  //CHECK:do {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} while ([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan10->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan10, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan10, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  do {
  } while (cufftExecZ2D(plan10, idata, odata));

  cufftHandle plan11;
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan11 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan11->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan11->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan11->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan11->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan11->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan11->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan11->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan11->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan11->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a switch statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:switch (int stat = 0){
  //CHECK-NEXT:}
  switch (int stat = cufftMakePlanMany(plan11, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)){
  }

  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan11->commit(q_ct1);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan11, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan11, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a switch statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:switch (int stat = 0){
  //CHECK-NEXT:}
  switch (int stat = cufftExecZ2D(plan11, idata, odata)){
  }
  return 0;
}

cufftResult foo1(cufftHandle plan) {
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a return statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:return 0;
  return cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo2(cufftHandle plan) {
  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(dpct::get_default_queue());
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, (double*)idata, odata);
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a return statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:return 0;
  return cufftExecZ2D(plan, idata, odata);
}

cufftResult foo3(cufftHandle plan) {
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo4(cufftHandle plan) {
  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(dpct::get_default_queue());
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, (double*)idata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, (double*)idata, odata);
  //CHECK-NEXT:}
  cufftExecZ2D(plan, idata, odata);
}

