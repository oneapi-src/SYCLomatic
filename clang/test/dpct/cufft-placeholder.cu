// RUN: c2s --format-range=none -out-root %T/cufft-placeholder %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
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

//CHECK:/*
//CHECK-NEXT:DPCT1050:{{[0-9]+}}: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
//CHECK-NEXT:*/
//CHECK-NEXT:void foo1(std::shared_ptr<oneapi::mkl::dft::descriptor<c2s_placeholder/*Fix the precision and domain type manually*/>> plan) {
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2* idata;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
//CHECK-NEXT:  */
//CHECK-NEXT:  plan->commit(c2s::get_default_queue());
//CHECK-NEXT:  if ((void *)idata == (void *)odata) {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, (double*)idata);
//CHECK-NEXT:  } else {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, (double*)idata, odata);
//CHECK-NEXT:  }
//CHECK-NEXT:}
void foo1(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

//CHECK:/*
//CHECK-NEXT:DPCT1050:{{[0-9]+}}: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
//CHECK-NEXT:*/
//CHECK-NEXT:void foo2(std::shared_ptr<oneapi::mkl::dft::descriptor<c2s_placeholder/*Fix the precision and domain type manually*/>> plan) {
//CHECK-NEXT:  float* odata;
//CHECK-NEXT:  sycl::float2* idata;
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
//CHECK-NEXT:  */
//CHECK-NEXT:  plan->commit(c2s::get_default_queue());
//CHECK-NEXT:  if ((void *)idata == (void *)odata) {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, (float*)idata);
//CHECK-NEXT:  } else {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, (float*)idata, odata);
//CHECK-NEXT:  }
//CHECK-NEXT:}
void foo2(cufftHandle plan) {
  float* odata;
  float2* idata;
  cufftExecC2R(plan, idata, odata);
}

int main() {
  //CHECK:/*
  //CHECK-NEXT:DPCT1050:{{[0-9]+}}: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:std::shared_ptr<oneapi::mkl::dft::descriptor<c2s_placeholder/*Fix the precision and domain type manually*/>> plan1;
  //CHECK-NEXT:int type1 = 108;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1068:{{[0-9]+}}: The value of dimensions and strides could not be deduced. You need to update 'c2s_placeholder' manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1050:{{[0-9]+}}: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1068:{{[0-9]+}}: The value of FFT type could not be deduced. You need to update 'FWD_DISTANCE' and 'BWD_DISTANCE' manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan1 = std::make_shared<oneapi::mkl::dft::descriptor<c2s_placeholder/*Fix the precision and domain type manually*/>>(c2s_placeholder/*Fix the dimensions manually*/);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 11);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[c2s_placeholder/*Fix the dimensions manually*/] = {c2s_placeholder/*Fix the stride manually*/};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[c2s_placeholder/*Fix the dimensions manually*/] = {c2s_placeholder/*Fix the stride manually*/};
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:}
  cufftHandle plan1;
  cufftType_t type1 = CUFFT_Z2D;
  cufftMakePlanMany(plan1, rank, n, inembed, istride, idist, onembed, ostride, odist, type1, 11, work_size);

  //CHECK:/*
  //CHECK-NEXT:DPCT1050:{{[0-9]+}}: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:std::shared_ptr<oneapi::mkl::dft::descriptor<c2s_placeholder/*Fix the precision and domain type manually*/>> plan2;
  //CHECK-NEXT:int type2 = 44;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1068:{{[0-9]+}}: The value of dimensions and strides could not be deduced. You need to update 'c2s_placeholder' manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The 'work_size' parameter could not be migrated. You may need to update the code manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1050:{{[0-9]+}}: The template argument of the FFT precision and domain type could not be deduced. You need to update this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function could not be deduced, so it is assumed out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1068:{{[0-9]+}}: The value of FFT type could not be deduced. You need to update 'FWD_DISTANCE' and 'BWD_DISTANCE' manually.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan2 = std::make_shared<oneapi::mkl::dft::descriptor<c2s_placeholder/*Fix the precision and domain type manually*/>>(c2s_placeholder/*Fix the dimensions manually*/);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[c2s_placeholder/*Fix the dimensions manually*/] = {c2s_placeholder/*Fix the stride manually*/};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[c2s_placeholder/*Fix the dimensions manually*/] = {c2s_placeholder/*Fix the stride manually*/};
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:}
  cufftHandle plan2;
  cufftType_t type2 = CUFFT_C2R;
  cufftMakePlanMany(plan2, rank, n, inembed, istride, idist, onembed, ostride, odist, type2, 12, work_size);

  foo1(plan1);
  foo2(plan2);

  return 0;
}

