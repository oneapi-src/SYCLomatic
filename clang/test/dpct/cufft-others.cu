// RUN: dpct --format-range=none -out-root %T/cufft-others %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/cufft-others/cufft-others.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>


int main() {
  //CHECK:dpct::fft::fft_engine_ptr plan;
  //CHECK-NEXT:sycl::float2* iodata;
  cufftHandle plan;
  float2* iodata;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  //CHECK:plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward);
  cufftExecR2C(plan, (float*)iodata, iodata);

  return 0;
}

int foo2() {
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
  //CHECK-NEXT:DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan_mmany64_Z2Z->commit(&dpct::get_default_queue(), 3, n_mmany64_Z2Z, inembed_mmany64_Z2Z, istride_mmany64_Z2Z, idist_mmany64_Z2Z, onembed_mmany64_Z2Z, ostride_mmany64_Z2Z, odist_mmany64_Z2Z, dpct::fft::fft_type::complex_double_to_complex_double, 12, work_size_mmany64_Z2Z);
  cufftMakePlanMany64(plan_mmany64_Z2Z, 3, n_mmany64_Z2Z, inembed_mmany64_Z2Z, istride_mmany64_Z2Z, idist_mmany64_Z2Z, onembed_mmany64_Z2Z, ostride_mmany64_Z2Z, odist_mmany64_Z2Z, CUFFT_Z2Z, 12, work_size_mmany64_Z2Z);

  //CHECK:plan_mmany64_Z2Z->compute<sycl::double2, sycl::double2>(idata_mmany64_Z2Z, odata_mmany64_Z2Z, dpct::fft::fft_direction::forward);
  cufftExecZ2Z(plan_mmany64_Z2Z, idata_mmany64_Z2Z, odata_mmany64_Z2Z, CUFFT_FORWARD);

  //CHECK:plan_mmany64_Z2Z->compute<sycl::double2, sycl::double2>(idata_mmany64_Z2Z, odata_mmany64_Z2Z, dpct::fft::fft_direction::backward);
  cufftExecZ2Z(plan_mmany64_Z2Z, idata_mmany64_Z2Z, odata_mmany64_Z2Z, CUFFT_INVERSE);

  return 0;
}

int foo3(cudaStream_t stream) {
  //CHECK:dpct::fft::fft_engine_ptr plan;
  //CHECK-NEXT:sycl::float2* iodata;
  cufftHandle plan;
  float2* iodata;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:plan->set_queue(stream);
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftSetStream(plan, stream);

  //CHECK:plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward);
  cufftExecR2C(plan, (float*)iodata, iodata);

  return 0;
}

void foo4(double x) {
  //CHECK:const int dir = -1;
  const int dir = CUFFT_FORWARD;
  cufftHandle plan;
  float2* iodata;
  cufftPlan1d(&plan, 10, CUFFT_C2C, 3);
  //CHECK:plan->compute<sycl::float2, sycl::float2>(iodata, iodata, dpct::fft::fft_direction::forward);
  //CHECK-NEXT:plan->compute<sycl::float2, sycl::float2>(iodata, iodata, dpct::fft::fft_direction::backward);
  cufftExecC2C(plan, iodata, iodata, dir);
  cufftExecC2C(plan, iodata, iodata, -dir);
  const double base = dir * 3.1415926 / x;
}

void foo5(double x) {
  //CHECK:int dir = -1;
  int dir = CUFFT_FORWARD;
  cufftHandle plan;
  float2* iodata;
  cufftPlan1d(&plan, 10, CUFFT_C2C, 3);
  //CHECK:plan->compute<sycl::float2, sycl::float2>(iodata, iodata, dir == 1 ? dpct::fft::fft_direction::backward : dpct::fft::fft_direction::forward);
  //CHECK-NEXT:plan->compute<sycl::float2, sycl::float2>(iodata, iodata, -dir == 1 ? dpct::fft::fft_direction::backward : dpct::fft::fft_direction::forward);
  cufftExecC2C(plan, iodata, iodata, dir);
  cufftExecC2C(plan, iodata, iodata, -dir);
  const double base = dir * 3.1415926 / x;
}
