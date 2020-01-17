// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types001.dp.cpp

#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cufft.h>

// CHECK: dpct::device_info deviceProp;
cudaDeviceProp deviceProp;

// CHECK: const dpct::device_info deviceProp1 = {};
const cudaDeviceProp deviceProp1 = {};

// CHECK: volatile dpct::device_info deviceProp2;
volatile cudaDeviceProp deviceProp2;

// CHDCK: sycl::event events[23];
cudaEvent_t events[23];
// CHECK: const sycl::event *pevents[23];
const cudaEvent_t *pevents[23];
// CHECK: const sycl::event **ppevents[23];
const cudaEvent_t **ppevents[23];

// CHECK: int errors[23];
cudaError_t errors[23];
// CHECK: const int *perrors[23];
const cudaError_t *perrors[23];
// CHECK: const int **pperrors[23];
const cudaError_t **pperrors[23];

// CHECK: int errors1[23];
cudaError errors1[23];
// CHECK: const int *perrors1[23];
const cudaError *perrors1[23];
// CHECK: const int **pperrors1[23];
const cudaError **pperrors1[23];

// CHECK: sycl::range<3> dims[23];
dim3 dims[23];
// CHECK: const sycl::range<3> *pdims[23];
const dim3 *pdims[23];
// CHECK: const sycl::range<3> **ppdims[23];
const dim3 **ppdims[23];

struct s {
  // CHECK: sycl::event events[23];
  cudaEvent_t events[23];
  // CHECK: const sycl::event *pevents[23];
  const cudaEvent_t *pevents[23];
  // CHECK: const sycl::event **ppevents[23];
  const cudaEvent_t **ppevents[23];

  // CHECK: int errors[23];
  cudaError_t errors[23];
  // CHECK: const int *perrors[23];
  const cudaError_t *perrors[23];
  // CHECK: const int **pperrors[23];
  const cudaError_t **pperrors[23];

  // CHECK: int errors1[23];
  cudaError errors1[23];
  // CHECK: const int *perrors1[23];
  const cudaError *perrors1[23];
  // CHECK: const int **pperrors1[23];
  const cudaError **pperrors1[23];

  // CHECK: sycl::range<3> dims[23];
  dim3 dims[23];
  // CHECK: const sycl::range<3> *pdims[23];
  const dim3 *pdims[23];
  // CHECK: const sycl::range<3> **ppdims[23];
  const dim3 **ppdims[23];
};

// CHECK:  void foo(dpct::device_info p) {
void foo(cudaDeviceProp p) {
  return;
}

// CHECK: int e;
cudaError e;

// CHECK: int ee;
cudaError_t ee;

// CHECK: int foo_0(int);
cudaError_t foo_0(cudaError_t);

// CHECK: int foo_1(int);
cudaError foo_1(cudaError_t);

// CHECK: int apicall(int i) {
cudaError_t apicall(int i) {
  return cudaSuccess;
};

// CHECK: int err = apicall(0);
cudaError_t err = apicall(0);

template <typename T>
// CHECK: void check(T result, char const *const func) {
void check(T result, char const *const func) {
}

#define checkCudaErrors(val) check((val), #val)

int main(int argc, char **argv) {
  //CHECK:sycl::range<3> d3(1, 1, 1);
  //CHECK-NEXT:int a = sizeof(sycl::range<3>);
  //CHECK-NEXT:a = sizeof(d3);
  //CHECK-NEXT:a = sizeof d3;
  dim3 d3;
  int a = sizeof(dim3);
  a = sizeof(d3);
  a = sizeof d3;

  //CHECK:int cudaErr_t;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(cudaErr_t);
  //CHECK-NEXT:a = sizeof cudaErr_t;
  cudaError_t cudaErr_t;
  a = sizeof(cudaError_t);
  a = sizeof(cudaErr_t);
  a = sizeof cudaErr_t;

  //CHECK:int res;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(res);
  //CHECK-NEXT:a = sizeof res;
  CUresult res;
  a = sizeof(CUresult);
  a = sizeof(res);
  a = sizeof res;

  //CHECK:void* context;
  //CHECK-NEXT:a = sizeof(void*);
  //CHECK-NEXT:a = sizeof(context);
  //CHECK-NEXT:a = sizeof context;
  CUcontext context;
  a = sizeof(CUcontext);
  a = sizeof(context);
  a = sizeof context;

  //CHECK:sycl::event event;
  //CHECK-NEXT:a = sizeof(sycl::event);
  //CHECK-NEXT:a = sizeof(event);
  //CHECK-NEXT:a = sizeof event;
  cudaEvent_t event;
  a = sizeof(cudaEvent_t);
  a = sizeof(event);
  a = sizeof event;

  //CHECK:queue_p stream;
  //CHECK-NEXT:a = sizeof(queue_p);
  //CHECK-NEXT:a = sizeof(stream);
  //CHECK-NEXT:a = sizeof stream;
  cudaStream_t stream;
  a = sizeof(cudaStream_t);
  a = sizeof(stream);
  a = sizeof stream;

  //CHECK:int cudaErr;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(cudaErr);
  //CHECK-NEXT:a = sizeof cudaErr;
  cudaError_t cudaErr;
  a = sizeof(cudaError_t);
  a = sizeof(cudaErr);
  a = sizeof cudaErr;

  //CHECK:sycl::half h;
  //CHECK-NEXT:a = sizeof(sycl::half);
  //CHECK-NEXT:a = sizeof(h);
  //CHECK-NEXT:a = sizeof h;
  half h;
  a = sizeof(half);
  a = sizeof(h);
  a = sizeof h;

  //CHECK:sycl::half2 h2;
  //CHECK-NEXT:a = sizeof(sycl::half2);
  //CHECK-NEXT:a = sizeof(h2);
  //CHECK-NEXT:a = sizeof h2;
  half2 h2;
  a = sizeof(half2);
  a = sizeof(h2);
  a = sizeof h2;

  //CHECK:int blasStatus;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(blasStatus);
  //CHECK-NEXT:a = sizeof blasStatus;
  cublasStatus_t blasStatus;
  a = sizeof(cublasStatus_t);
  a = sizeof(blasStatus);
  a = sizeof blasStatus;

  //CHECK:sycl::float2 complex;
  //CHECK-NEXT:a = sizeof(sycl::float2);
  //CHECK-NEXT:a = sizeof(complex);
  //CHECK-NEXT:a = sizeof complex;
  cuComplex complex;
  a = sizeof(cuComplex);
  a = sizeof(complex);
  a = sizeof complex;

  //CHECK:sycl::double2 doubleComplex;
  //CHECK-NEXT:a = sizeof(sycl::double2);
  //CHECK-NEXT:a = sizeof(doubleComplex);
  //CHECK-NEXT:a = sizeof doubleComplex;
  cuDoubleComplex doubleComplex;
  a = sizeof(cuDoubleComplex);
  a = sizeof(doubleComplex);
  a = sizeof doubleComplex;

  //CHECK:mkl::uplo fill;
  //CHECK-NEXT:a = sizeof(mkl::uplo);
  //CHECK-NEXT:a = sizeof(fill);
  //CHECK-NEXT:a = sizeof fill;
  cublasFillMode_t fill;
  a = sizeof(cublasFillMode_t);
  a = sizeof(fill);
  a = sizeof fill;

  //CHECK:mkl::diag diag;
  //CHECK-NEXT:a = sizeof(mkl::diag);
  //CHECK-NEXT:a = sizeof(diag);
  //CHECK-NEXT:a = sizeof diag;
  cublasDiagType_t diag;
  a = sizeof(cublasDiagType_t);
  a = sizeof(diag);
  a = sizeof diag;

  //CHECK:mkl::side side;
  //CHECK-NEXT:a = sizeof(mkl::side);
  //CHECK-NEXT:a = sizeof(side);
  //CHECK-NEXT:a = sizeof side;
  cublasSideMode_t side;
  a = sizeof(cublasSideMode_t);
  a = sizeof(side);
  a = sizeof side;

  //CHECK:mkl::transpose oper;
  //CHECK-NEXT:a = sizeof(mkl::transpose);
  //CHECK-NEXT:a = sizeof(oper);
  //CHECK-NEXT:a = sizeof oper;
  cublasOperation_t oper;
  a = sizeof(cublasOperation_t);
  a = sizeof(oper);
  a = sizeof oper;

  //CHECK:int blasStatus_legacy;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(blasStatus_legacy);
  //CHECK-NEXT:a = sizeof blasStatus_legacy;
  cublasStatus blasStatus_legacy;
  a = sizeof(cublasStatus);
  a = sizeof(blasStatus_legacy);
  a = sizeof blasStatus_legacy;

  //CHECK:int solverStatus;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(solverStatus);
  //CHECK-NEXT:a = sizeof solverStatus;
  cusolverStatus_t solverStatus;
  a = sizeof(cusolverStatus_t);
  a = sizeof(solverStatus);
  a = sizeof solverStatus;

  //CHECK:int64_t eigtype;
  //CHECK-NEXT:a = sizeof(int64_t);
  //CHECK-NEXT:a = sizeof(eigtype);
  //CHECK-NEXT:a = sizeof eigtype;
  cusolverEigType_t eigtype;
  a = sizeof(cusolverEigType_t);
  a = sizeof(eigtype);
  a = sizeof eigtype;

  //CHECK:mkl::job eigmode;
  //CHECK-NEXT:a = sizeof(mkl::job);
  //CHECK-NEXT:a = sizeof(eigmode);
  //CHECK-NEXT:a = sizeof eigmode;
  cusolverEigMode_t eigmode;
  a = sizeof(cusolverEigMode_t);
  a = sizeof(eigmode);
  a = sizeof eigmode;

  //CHECK:int randstatus_t;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(randstatus_t);
  //CHECK-NEXT:a = sizeof randstatus_t;
  curandStatus_t randstatus_t;
  a = sizeof(curandStatus_t);
  a = sizeof(randstatus_t);
  a = sizeof randstatus_t;

  //CHECK:int cudaerror;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(cudaerror);
  //CHECK-NEXT:a = sizeof cudaerror;
  cudaError cudaerror;
  a = sizeof(cudaError);
  a = sizeof(cudaerror);
  a = sizeof cudaerror;

  //CHECK:int fftresult;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(fftresult);
  //CHECK-NEXT:a = sizeof fftresult;
  cufftResult_t fftresult;
  a = sizeof(cufftResult_t);
  a = sizeof(fftresult);
  a = sizeof fftresult;

  //CHECK:cudaError_enum error_enum;
  //CHECK-NEXT:a = sizeof(cudaError_enum);
  //CHECK-NEXT:a = sizeof(error_enum);
  //CHECK-NEXT:a = sizeof error_enum;
  cudaError_enum error_enum;
  a = sizeof(cudaError_enum);
  a = sizeof(error_enum);
  a = sizeof error_enum;

  //CHECK:int randstatus;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(randstatus);
  //CHECK-NEXT:a = sizeof randstatus;
  curandStatus randstatus;
  a = sizeof(curandStatus);
  a = sizeof(randstatus);
  a = sizeof randstatus;

  //CHECK:dpct::device_info dp;
  //CHECK-NEXT:a = sizeof(dpct::device_info);
  //CHECK-NEXT:a = sizeof(dp);
  //CHECK-NEXT:a = sizeof dp;
  cudaDeviceProp dp;
  a = sizeof(cudaDeviceProp);
  a = sizeof(dp);
  a = sizeof dp;

  //CHECK:CUstream_st *stream_st;
  //CHECK-NEXT:a = sizeof(CUstream_st*);
  //CHECK-NEXT:a = sizeof(stream_st);
  //CHECK-NEXT:a = sizeof stream_st;
  CUstream_st *stream_st;
  a = sizeof(CUstream_st*);
  a = sizeof(stream_st);
  a = sizeof stream_st;

  //CHECK:CUevent_st *event_st;
  //CHECK-NEXT:a = sizeof(CUevent_st*);
  //CHECK-NEXT:a = sizeof(event_st);
  //CHECK-NEXT:a = sizeof event_st;
  CUevent_st *event_st;
  a = sizeof(CUevent_st*);
  a = sizeof(event_st);
  a = sizeof event_st;

  //CHECK:sycl::queue blashandle;
  //CHECK-NEXT:a = sizeof(sycl::queue);
  //CHECK-NEXT:a = sizeof(blashandle);
  //CHECK-NEXT:a = sizeof blashandle;
  cublasHandle_t blashandle;
  a = sizeof(cublasHandle_t);
  a = sizeof(blashandle);
  a = sizeof blashandle;

  //CHECK:sycl::queue solverdnhandle;
  //CHECK-NEXT:a = sizeof(sycl::queue);
  //CHECK-NEXT:a = sizeof(solverdnhandle);
  //CHECK-NEXT:a = sizeof solverdnhandle;
  cusolverDnHandle_t solverdnhandle;
  a = sizeof(cusolverDnHandle_t);
  a = sizeof(solverdnhandle);
  a = sizeof solverdnhandle;

  checkCudaErrors(apicall(0));
  return 0;
}

__global__ void foo() {
  void *p;
  // CHECK: (queue_p)p;
  // CHECK-NEXT: (queue_p *)p;
  // CHECK-NEXT: (queue_p **)p;
  // CHECK-NEXT: (queue_p ***)p;
  (cudaStream_t)p;
  (cudaStream_t *)p;
  (cudaStream_t **)p;
  (cudaStream_t ***)p;

  int i;
  // CHECK: (int)i;
  // CHECK-NEXT: (int *)p;
  // CHECK-NEXT: (int **)p;
  // CHECK-NEXT: (int ***)p;
  (cudaError)i;
  (cudaError *)p;
  (cudaError **)p;
  (cudaError ***)p;

  cudaDeviceProp cdp;
  // CHECK: dpct::device_info cdp2 = (dpct::device_info)cdp;
  // CHECK-NEXT: (dpct::device_info *)p;
  // CHECK-NEXT: (dpct::device_info **)p;
  // CHECK-NEXT: (dpct::device_info ***)p;
  cudaDeviceProp cdp2 = (cudaDeviceProp)cdp;
  (cudaDeviceProp *)p;
  (cudaDeviceProp **)p;
  (cudaDeviceProp ***)p;
}

