// RUN: dpct --format-range=none --usm-level=none -out-root %T/cufft-different-locations %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-different-locations/cufft-different-locations.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>

cufftHandle plan;
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
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int res1 = (plan->commit(q_ct1), 0);
  cufftResult res1 = cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  //CHECK:int res2 = 0;
  //CHECK-NEXT:{
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:res2 = (oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}}), 0);
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  cufftResult res2 = cufftExecZ2D(plan, idata, odata);

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:res1 = (plan->commit(q_ct1), 0);
  res1 = cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
  //CHECK:{
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:res2 = (oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}}), 0);
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  res2 = cufftExecZ2D(plan, idata, odata);

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(q_ct1);
  HANDLE_CUFFT_ERROR(cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  //CHECK:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in function-like macro statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:HANDLE_CUFFT_ERROR(0);
  //CHECK:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:HANDLE_CUFFT_ERROR([&](){
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  HANDLE_CUFFT_ERROR(cufftExecZ2D(plan, idata, odata));

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(0) {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} else if ([&](){
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}()) {
  //CHECK-NEXT:}
  if(cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  } else if (cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  //CHECK:{
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if (0) {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} else if([&](){
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}()) {
  //CHECK-NEXT:}
  if (cufftExecZ2D(plan, idata, odata)) {
  } else if(cufftExecZ2D(plan, idata, odata)) {
  }

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(int res = 0) {
  //CHECK-NEXT:}
  if(cufftResult res = cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)) {
  }
  //CHECK:{
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in an if statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:if(int res = 0) {
  //CHECK-NEXT:}
  if(cufftResult res = cufftExecZ2D(plan, idata, odata)) {
  }

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a for statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (0;;) {
  //CHECK-NEXT:}
  for (cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);;) {
  }
  //CHECK:{
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a for statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (0;;) {
  //CHECK-NEXT:}
  for (cufftExecZ2D(plan, idata, odata);;) {
  }


  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (;[&](){
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}();) {
  //CHECK-NEXT:}
  for (;cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);) {
  }
  //CHECK:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:for (;[&](){
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}();) {
  //CHECK-NEXT:}
  for (;cufftExecZ2D(plan, idata, odata);) {
  }

  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:while ([&](){
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}() != 0) {
  //CHECK-NEXT:}
  while (cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size) != 0) {
  }
  //CHECK:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:while ([&](){
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}() != 0) {
  //CHECK-NEXT:}
  while (cufftExecZ2D(plan, idata, odata) != 0) {
  }


  //CHECK:do {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} while ([&](){
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  do {
  } while (cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size));
  //CHECK:do {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:} while ([&](){
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  do {
  } while (cufftExecZ2D(plan, idata, odata));


  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a switch statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:switch (int stat = 0){
  //CHECK-NEXT:}
  switch (int stat = cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size)){
  }
  //CHECK:{
  //CHECK-NEXT:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a switch statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:switch (int stat = 0){
  //CHECK-NEXT:}
  switch (int stat = cufftExecZ2D(plan, idata, odata)){
  }
  return 0;
}

cufftResult foo1() {
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(dpct::get_default_queue());
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a return statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:return 0;
  return cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo2() {
  //CHECK:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(dpct::get_default_queue());
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors, it does not use error codes. 0 is used instead of an error code in a return statement. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:return 0;
  return cufftExecZ2D(plan, idata, odata);
}

cufftResult foo3() {
  //CHECK:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan->commit(dpct::get_default_queue());
  cufftMakePlanMany(plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, 12, work_size);
}

cufftResult foo4() {
  //CHECK:auto idata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(idata);
  //CHECK-NEXT:if ((void *)idata == (void *)odata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan->commit(dpct::get_default_queue());
  //CHECK-NEXT:  auto odata_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(odata);
  //CHECK-NEXT:oneapi::mkl::dft::compute_backward(*plan, idata_buf_ct{{[0-9]+}}, odata_buf_ct{{[0-9]+}});
  //CHECK-NEXT:}
  cufftExecZ2D(plan, idata, odata);
}

