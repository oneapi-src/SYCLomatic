// RUN: dpct --format-range=none -out-root %T/cufft-others %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/cufft-others/cufft-others.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>


int main() {
  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>> plan;
  //CHECK-NEXT:sycl::float2* iodata;
  cufftHandle plan;
  float2* iodata;

  //CHECK:plan = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(10 + 2);
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[2] = {0, 1};
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, ((10 + 2)/2+1)*2);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, (10 + 2)/2+1);
  //CHECK-NEXT:plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 3);
  //CHECK-NEXT:plan->commit(dpct::get_default_queue());
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  //CHECK:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  cufftExecR2C(plan, (float*)iodata, iodata);

  return 0;
}

