// RUN: dpct --format-range=none -out-root %T/cufft-refine-setstream %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-refine-setstream/cufft-refine-setstream.dp.cpp --match-full-lines %s
#include "cufft.h"

void foo1() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:plan->set_queue(s);
  //CHECK-NEXT:plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward);
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftSetStream(plan, s);
  cufftExecR2C(plan, (float*)iodata, iodata);
}


#define cufftCheck(stmt) \
do {                                           \
  cufftResult err = stmt;                                               \
  if (err != CUFFT_SUCCESS) {                                           \
  }                                                                     \
} while(0)
void foo2() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR(plan->set_queue(s)));
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftCheck(cufftSetStream(plan, s));
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
}
#undef cufftCheck



#define HANDLE_CUFFT_ERROR( err ) (CufftHandleError( err, __FILE__, __LINE__ ))
static void CufftHandleError( cufftResult err, const char *file, int line )
{
    if (err != CUFFT_SUCCESS)
    {
      int a = err;
    }
}

void foo3() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:HANDLE_CUFFT_ERROR(DPCT_CHECK_ERROR(plan->set_queue(s)));
  //CHECK-NEXT:HANDLE_CUFFT_ERROR(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  HANDLE_CUFFT_ERROR(cufftSetStream(plan, s));
  HANDLE_CUFFT_ERROR(cufftExecR2C(plan, (float*)iodata, iodata));
}
#undef HANDLE_CUFFT_ERROR


void foo4() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:int err = DPCT_CHECK_ERROR(plan->set_queue(s));
  //CHECK-NEXT:err = DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward)));
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftResult err = cufftSetStream(plan, s);
  err = cufftExecR2C(plan, (float*)iodata, iodata);
}


static inline void CUFFT_CHECK(cufftResult error)
{
  if (error != CUFFT_SUCCESS) {
  }
}

void foo5() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:CUFFT_CHECK(DPCT_CHECK_ERROR(plan->set_queue(s)));
  //CHECK-NEXT:CUFFT_CHECK(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  CUFFT_CHECK(cufftSetStream(plan, s));
  CUFFT_CHECK(cufftExecR2C(plan, (float*)iodata, iodata));
}


#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)
template <typename T>
void my_error_checker(T ReturnValue, char const *const FuncName) {}

void foo6() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:MY_ERROR_CHECKER(DPCT_CHECK_ERROR(plan->set_queue(s)));
  //CHECK-NEXT:MY_ERROR_CHECKER(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  MY_ERROR_CHECKER(cufftSetStream(plan, s));
  MY_ERROR_CHECKER(cufftExecR2C(plan, (float*)iodata, iodata));
}
#undef MY_ERROR_CHECKER


#define CHECK_CUFFT(call)                                                      \
{                                                                              \
  cufftResult err;                                                           \
  if ( (err = (call)) != CUFFT_SUCCESS)                                      \
  {                                                                          \
  }                                                                          \
}
void foo7() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:CHECK_CUFFT(DPCT_CHECK_ERROR(plan->set_queue(s)));
  //CHECK-NEXT:CHECK_CUFFT(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  CHECK_CUFFT(cufftSetStream(plan, s));
  CHECK_CUFFT(cufftExecR2C(plan, (float*)iodata, iodata));
}
#undef CHECK_CUFFT

#define cufftCheck(stmt) \
do {                                           \
  cufftResult err;                                                      \
  if ( (err = (stmt)) != CUFFT_SUCCESS) {                               \
  }                                                                     \
} while(0)
void foo8() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR(plan->set_queue(s)));
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftCheck(cufftSetStream(plan, s));
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
}


void foo9() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s1, s2;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR(plan->set_queue(s1)));
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR(plan->set_queue(s2)));
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
  cufftCheck(cufftSetStream(plan, s1));
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
  cufftCheck(cufftSetStream(plan, s2));
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
}

// This case needs manual fix
void foo10(bool flag) {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  //CHECK-NEXT:if (flag) {
  //CHECK-NEXT:  cufftCheck(DPCT_CHECK_ERROR(plan->set_queue(s)));
  //CHECK-NEXT:}
  //CHECK-NEXT:cufftCheck(DPCT_CHECK_ERROR((plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward))));
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
  if (flag) {
    cufftCheck(cufftSetStream(plan, s));
  }
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
}
#undef cufftCheck

void foo11(bool flag) {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:if (flag)
  //CHECK-NEXT:  plan->set_queue(s);
  //CHECK-NEXT:plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward);
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  if (flag)
    cufftSetStream(plan, s);
  cufftExecR2C(plan, (float*)iodata, iodata);
}

void foo12(cufftHandle plan2) {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:plan->set_queue(s);
  //CHECK-NEXT:plan = plan2;
  //CHECK-NEXT:plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward);
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftSetStream(plan, s);
  plan = plan2;
  cufftExecR2C(plan, (float*)iodata, iodata);
}

void changeHandle(cufftHandle &p);

void foo13() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:plan->set_queue(s);
  //CHECK-NEXT:changeHandle(plan);
  //CHECK-NEXT:plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward);
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftSetStream(plan, s);
  changeHandle(plan);
  cufftExecR2C(plan, (float*)iodata, iodata);
}

void foo14() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  //CHECK:plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), 10 + 2, dpct::fft::fft_type::real_float_to_complex_float, 3);
  //CHECK-NEXT:plan->set_queue(s);
  //CHECK-NEXT:plan = dpct::fft::fft_engine::create();
  //CHECK-NEXT:plan->compute<float, sycl::float2>((float*)iodata, iodata, dpct::fft::fft_direction::forward);
  //CHECK-NEXT:dpct::fft::fft_engine::destroy(plan);
  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftSetStream(plan, s);
  cufftCreate(&plan);
  cufftExecR2C(plan, (float*)iodata, iodata);
  cufftDestroy(plan);
}
