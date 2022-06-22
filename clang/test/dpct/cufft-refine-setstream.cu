// RUN: dpct --format-range=none -out-root %T/cufft-refine-setstream %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cufft-refine-setstream/cufft-refine-setstream.dp.cpp --match-full-lines %s
#include "cufft.h"

void foo1() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftSetStream(plan, s);

  //CHECK:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
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

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftCheck(cufftSetStream(plan, s));

  //CHECK:cufftCheck([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
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

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  HANDLE_CUFFT_ERROR(cufftSetStream(plan, s));

  //CHECK:HANDLE_CUFFT_ERROR([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  HANDLE_CUFFT_ERROR(cufftExecR2C(plan, (float*)iodata, iodata));
}
#undef HANDLE_CUFFT_ERROR


void foo4() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftResult err = cufftSetStream(plan, s);

  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:err = (oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata), 0);
  //CHECK-NEXT:}
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

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  CUFFT_CHECK(cufftSetStream(plan, s));

  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:CUFFT_CHECK((oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata), 0));
  CUFFT_CHECK(cufftExecR2C(plan, (float*)iodata, iodata));
}


#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)
template <typename T>
void my_error_checker(T ReturnValue, char const *const FuncName) {}

void foo6() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  MY_ERROR_CHECKER(cufftSetStream(plan, s));

  //CHECK:MY_ERROR_CHECKER([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
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

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  CHECK_CUFFT(cufftSetStream(plan, s));

  //CHECK:CHECK_CUFFT([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
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

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);
  cufftCheck(cufftSetStream(plan, s));

  //CHECK:cufftCheck([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
}


void foo9() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s1, s2;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  //CHECK:cufftCheck([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
  
  cufftCheck(cufftSetStream(plan, s1));

  //CHECK:cufftCheck([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s1);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));

  cufftCheck(cufftSetStream(plan, s2));

  //CHECK:cufftCheck([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s2);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
}

// This case needs manual fix
void foo10(bool flag) {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  //CHECK:cufftCheck([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));

  if (flag) {
    cufftCheck(cufftSetStream(plan, s));
  }

  //CHECK:cufftCheck([&](){
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(q_ct1);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  //CHECK-NEXT:return 0;
  //CHECK-NEXT:}());
  cufftCheck(cufftExecR2C(plan, (float*)iodata, iodata));
}
#undef cufftCheck

void foo11(bool flag) {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  if (flag)
    cufftSetStream(plan, s);

  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(dpct::get_default_queue());
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  cufftExecR2C(plan, (float*)iodata, iodata);
}

void foo12(cufftHandle plan2) {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  cufftSetStream(plan, s);
  plan = plan2;

  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  cufftExecR2C(plan, (float*)iodata, iodata);
}

void changeHandle(cufftHandle &p);

void foo13() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  cufftSetStream(plan, s);
  changeHandle(plan);

  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  cufftExecR2C(plan, (float*)iodata, iodata);
}

void foo14() {
  cufftHandle plan;
  float2* iodata;
  cudaStream_t s;

  cufftPlan1d(&plan, 10 + 2, CUFFT_R2C, 3);

  cufftSetStream(plan, s);
  //CHECK: DPCT1026:{{[0-9]+}}: The call to cufftCreate was removed because this call is redundant in SYCL.
  cufftCreate(&plan);

  //CHECK:/*
  //CHECK-NEXT:DPCT1075:{{[0-9]+}}: Migration of cuFFT calls may be incorrect and require review.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan->commit(*s);
  //CHECK-NEXT:if ((void *)(float*)iodata == (void *)iodata) {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata);
  //CHECK-NEXT:} else {
  //CHECK-NEXT:oneapi::mkl::dft::compute_forward(*plan, (float*)iodata, (float*)iodata);
  //CHECK-NEXT:}
  cufftExecR2C(plan, (float*)iodata, iodata);
  //CHECK: DPCT1026:{{[0-9]+}}: The call to cufftDestroy was removed because this call is redundant in SYCL.
  cufftDestroy(plan);
}
