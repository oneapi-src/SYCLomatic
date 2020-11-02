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

//CHECK:void foo1(std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan) {
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2* idata;
//CHECK-NEXT:  if ((void *)idata == (void *)odata) {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, idata);
//CHECK-NEXT:  } else {
//CHECK-NEXT:  plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
//CHECK-NEXT:  plan->commit(dpct::get_default_queue());
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, idata, odata);
//CHECK-NEXT:  }
//CHECK-NEXT:}
void foo1(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

//CHECK:void foo2(std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan) {
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2* idata;
//CHECK-NEXT:  if ((void *)idata == (void *)odata) {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, idata);
//CHECK-NEXT:  } else {
//CHECK-NEXT:  plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
//CHECK-NEXT:  plan->commit(dpct::get_default_queue());
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, idata, odata);
//CHECK-NEXT:  }
//CHECK-NEXT:}
void foo2(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

int main() {
  //CHECK:constexpr int type = 108;
  constexpr cufftType_t type = CUFFT_Z2D;

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan1;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan1 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan1->commit(q_ct1);
  cufftHandle plan1;
  cufftMakePlanMany(plan1, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan2;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1066:{{[0-9]+}}: Migration is supported only if the input distance is the same as the output distance. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan2 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, idist);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, std::array<std::int64_t, 4>{0, istride, inembed[2] * istride, inembed[2] * inembed[1] * istride});
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, std::array<std::int64_t, 4>{0, ostride, onembed[2] * ostride, onembed[2] * onembed[1] * ostride});
  //CHECK-NEXT:plan2->commit(q_ct1);
  cufftHandle plan2;
  cufftMakePlanMany(plan2, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);

  foo1(plan1);
  foo2(plan2);

  return 0;
}

