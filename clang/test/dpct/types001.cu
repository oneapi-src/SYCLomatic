// RUN: dpct -out-root %T/types001 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck %s --match-full-lines --input-file %T/types001/types001.dp.cpp

#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cufft.h>
#include <stdio.h>
#include <vector>

// CHECK: dpct::device_info deviceProp;
cudaDeviceProp deviceProp;

// CHECK: const dpct::device_info deviceProp1 = {};
const cudaDeviceProp deviceProp1 = {};

// CHECK: volatile dpct::device_info deviceProp2;
volatile cudaDeviceProp deviceProp2;

// CHDCK: dpct::event_ptr events[23];
cudaEvent_t events[23];
// CHECK: const dpct::event_ptr *pevents[23];
const cudaEvent_t *pevents[23];
// CHECK: const dpct::event_ptr **ppevents[23];
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
  // CHECK: dpct::event_ptr events[23];
  cudaEvent_t events[23];
  // CHECK: const dpct::event_ptr *pevents[23];
  const cudaEvent_t *pevents[23];
  // CHECK: const dpct::event_ptr **ppevents[23];
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
// CHECK: void my_error_checker(T ReturnValue, char const *const FuncName) {
void my_error_checker(T ReturnValue, char const *const FuncName) {
}

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)

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

  //CHECK:int context;
  //CHECK-NEXT:a = sizeof(int);
  //CHECK-NEXT:a = sizeof(context);
  //CHECK-NEXT:a = sizeof context;
  CUcontext context;
  a = sizeof(CUcontext);
  a = sizeof(context);
  a = sizeof context;

  //CHECK:dpct::event_ptr event;
  //CHECK-NEXT:a = sizeof(dpct::event_ptr);
  //CHECK-NEXT:a = sizeof(event);
  //CHECK-NEXT:a = sizeof event;
  cudaEvent_t event;
  a = sizeof(cudaEvent_t);
  a = sizeof(event);
  a = sizeof event;

  //CHECK:dpct::queue_ptr stream;
  //CHECK-NEXT:a = sizeof(dpct::queue_ptr);
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

  //CHECK:oneapi::mkl::uplo fill;
  //CHECK-NEXT:a = sizeof(oneapi::mkl::uplo);
  //CHECK-NEXT:a = sizeof(fill);
  //CHECK-NEXT:a = sizeof fill;
  cublasFillMode_t fill;
  a = sizeof(cublasFillMode_t);
  a = sizeof(fill);
  a = sizeof fill;

  //CHECK:oneapi::mkl::diag diag;
  //CHECK-NEXT:a = sizeof(oneapi::mkl::diag);
  //CHECK-NEXT:a = sizeof(diag);
  //CHECK-NEXT:a = sizeof diag;
  cublasDiagType_t diag;
  a = sizeof(cublasDiagType_t);
  a = sizeof(diag);
  a = sizeof diag;

  //CHECK:oneapi::mkl::side side;
  //CHECK-NEXT:a = sizeof(oneapi::mkl::side);
  //CHECK-NEXT:a = sizeof(side);
  //CHECK-NEXT:a = sizeof side;
  cublasSideMode_t side;
  a = sizeof(cublasSideMode_t);
  a = sizeof(side);
  a = sizeof side;

  //CHECK:oneapi::mkl::transpose oper;
  //CHECK-NEXT:a = sizeof(oneapi::mkl::transpose);
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

  //CHECK:oneapi::mkl::job eigmode;
  //CHECK-NEXT:a = sizeof(oneapi::mkl::job);
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

  //CHECK:sycl::queue *stream_st;
  //CHECK-NEXT:a = sizeof(sycl::queue *);
  //CHECK-NEXT:a = sizeof(stream_st);
  //CHECK-NEXT:a = sizeof stream_st;
  CUstream_st *stream_st;
  a = sizeof(CUstream_st*);
  a = sizeof(stream_st);
  a = sizeof stream_st;

  //CHECK:sycl::event *event_st;
  //CHECK-NEXT:a = sizeof(sycl::event *);
  //CHECK-NEXT:a = sizeof(event_st);
  //CHECK-NEXT:a = sizeof event_st;
  CUevent_st *event_st;
  a = sizeof(CUevent_st*);
  a = sizeof(event_st);
  a = sizeof event_st;

  //CHECK:sycl::queue *blashandle;
  //CHECK-NEXT:a = sizeof(sycl::queue *);
  //CHECK-NEXT:a = sizeof(blashandle);
  //CHECK-NEXT:a = sizeof blashandle;
  cublasHandle_t blashandle;
  a = sizeof(cublasHandle_t);
  a = sizeof(blashandle);
  a = sizeof blashandle;

  //CHECK:sycl::queue *solverdnhandle;
  //CHECK-NEXT:a = sizeof(sycl::queue *);
  //CHECK-NEXT:a = sizeof(solverdnhandle);
  //CHECK-NEXT:a = sizeof solverdnhandle;
  cusolverDnHandle_t solverdnhandle;
  a = sizeof(cusolverDnHandle_t);
  a = sizeof(solverdnhandle);
  a = sizeof solverdnhandle;

  MY_ERROR_CHECKER(apicall(0));
  return 0;
}

__global__ void foo() {
  void *p;
  // CHECK: (dpct::queue_ptr) p;
  // CHECK-NEXT: (dpct::queue_ptr *)p;
  // CHECK-NEXT: (dpct::queue_ptr **)p;
  // CHECK-NEXT: (dpct::queue_ptr ***)p;
  (cudaStream_t)p;
  (cudaStream_t *)p;
  (cudaStream_t **)p;
  (cudaStream_t ***)p;


  // CHECK: malloc(sizeof(dpct::queue_ptr *));
  // CHECK-NEXT: malloc(sizeof(dpct::queue_ptr **));
  // CHECK-NEXT: malloc(sizeof(dpct::queue_ptr ***));
  // CHECK-NEXT: malloc(sizeof(dpct::queue_ptr &));
  malloc(sizeof(cudaStream_t *));
  malloc(sizeof(cudaStream_t **));
  malloc(sizeof(cudaStream_t ***));
  malloc(sizeof(cudaStream_t &));

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

template <typename T> struct S {};

// CHECK: template <> struct S<dpct::queue_ptr> {};
// CHECK-NEXT: template <> struct S<sycl::queue> {};
// CHECK-NEXT: template <> struct S<sycl::float2> {};
// CHECK-NEXT: template <> struct S<sycl::float4> {};
template <> struct S<cudaStream_t> {};
template <> struct S<CUstream_st> {};
template <> struct S<float2> {};
template <> struct S<float4> {};

void foobar() {
  // CHECK: S<dpct::queue_ptr> s0;
  S<cudaStream_t> s0;
  // CHECK: S<sycl::float2> s1;
  S<float2> s1;
  // CHECK: S<sycl::float4> s2;
  S<float4> s2;
}

void fun() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: dpct::queue_ptr *p, s, &r = s;
  cudaStream_t *p, s, &r = s;
  // CHECK: dpct::queue_ptr const s_2 = &q_ct1, *p_2, &r_2 = s;
  cudaStream_t const s_2 = NULL, *p_2, &r_2 = s;
  // CHECK: const dpct::queue_ptr &r_3 = s, *p_3, s_3 = &q_ct1;
  const cudaStream_t &r_3 = s, *p_3, s_3 = NULL;

  // CHECK: dpct::queue_ptr const *pc, sc = s, &rc = s;
  cudaStream_t const *pc, sc = s, &rc = s;
  // CHECK: const dpct::queue_ptr *pc1, sc1 = s, &rc1 = s;
  const cudaStream_t *pc1, sc1 = s, &rc1 = s;
  // CHECK: dpct::queue_ptr s1, *p1, &r1 = *p1;
  cudaStream_t s1, *p1, &r1 = *p1;
  // CHECK: dpct::queue_ptr &r2 = s1, *p2, s2;
  cudaStream_t &r2 = s1, *p2, s2;

  // CHECK: dpct::queue_ptr &r3 = s2,
  // CHECK-NEXT:             *p3,
  // CHECK-NEXT:             s3;
  cudaStream_t &r3 = s2,
               *p3,
               s3;

  // CHECK: dpct::queue_ptr const s4 = s1, s5 = s2;
  cudaStream_t const s4 = s1, s5 = s2;
  // CHECK: const dpct::queue_ptr s6 = s1, s7 = s2;
  const cudaStream_t s6 = s1, s7 = s2;

  // CHECK: dpct::queue_ptr const *s8, *s9;
  cudaStream_t const *s8, *s9;
  // CHECK: const dpct::queue_ptr *s10, *s11;
  const cudaStream_t *s10, *s11;
  // CHECK: dpct::queue_ptr *const s12 = &q_ct1, *const s13 = &q_ct1;
  cudaStream_t *const s12 = NULL, *const s13 = NULL;
  // CHECK: const dpct::queue_ptr *const s14 = &q_ct1, *const s15 = &q_ct1;
  const cudaStream_t *const s14 = NULL, *const s15 = NULL;
}

void fun2() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: dpct::queue_ptr s, s2;
  cudaStream_t s, s2;
  // CHECK: dpct::queue_ptr const s3 = &q_ct1, s4 = &q_ct1;
  cudaStream_t const s3 = NULL, s4 = NULL;
  // CHECK: const dpct::queue_ptr s5 = &q_ct1, s6 = &q_ct1;
  const cudaStream_t s5 = NULL, s6 = NULL;

  // CHECK: dpct::queue_ptr *s7, *const s8 = &q_ct1;
  cudaStream_t *s7, *const s8 = NULL;
  // CHECK: dpct::queue_ptr *const s9 = &q_ct1, *s10;
  cudaStream_t *const s9 = NULL, *s10;
  // CHECK: const dpct::queue_ptr *s11, *const s12 = &q_ct1;
  const cudaStream_t *s11, *const s12 = NULL;
  // CHECK: dpct::queue_ptr const *const s13 = &q_ct1, *s14;
  cudaStream_t const *const s13 = NULL, *s14;
  // CHECK: const dpct::queue_ptr *const s15 = &q_ct1, *s16;
  const cudaStream_t *const s15 = NULL, *s16;
  // CHECK: dpct::queue_ptr const *s17, *const s18 = &q_ct1;
  cudaStream_t const *s17, *const s18 = NULL;
}

// CHECK:template <>
// CHECK-NEXT:struct S<int &&> {};
// CHECK-NEXT:template <> struct S<int> {};
// CHECK-NEXT:template <> struct S<int *> {};
// CHECK-NEXT:template <> struct S<int &> {};
// CHECK-NEXT:template <> struct S<int &&> {};
template <>
struct S<int &&> {};
template <> struct S<cudaError_t> {};
template <> struct S<cudaError_t *> {};
template <> struct S<cudaError_t &> {};
template <> struct S<cudaError_t &&> {};

// CHECK: template <int SMEM_CONFIG = 0>
// CHECK-NEXT: class BlockRadixRank0 {};
// CHECK-NEXT: template <int SMEM_CONFIG = 1>
// CHECK-NEXT: class BlockRadixRank1 {};
// CHECK-NEXT: template <int SMEM_CONFIG = 2>
// CHECK-NEXT: class BlockRadixRank2 {};
template <cudaSharedMemConfig SMEM_CONFIG = cudaSharedMemBankSizeDefault>
class BlockRadixRank0 {};
template <cudaSharedMemConfig SMEM_CONFIG = cudaSharedMemBankSizeFourByte>
class BlockRadixRank1 {};
template <cudaSharedMemConfig SMEM_CONFIG = cudaSharedMemBankSizeEightByte>
class BlockRadixRank2 {};


void fun3() {
  char devstr[128] = "";
  // CHECK: dpct::device_info deviceProp;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1051:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with pciDomainID. It was migrated to -1. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1051:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with pciBusID. It was migrated to -1. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1051:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with pciDeviceID. It was migrated to -1. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sprintf(devstr, "pci %x:%x:%x", -1, -1, -1);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1051:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with concurrentKernels. It was migrated to true. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (true) {
  // CHECK-NEXT: }
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1051:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with canMapHostMemory. It was migrated to false. You may need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if (!false) {
  // CHECK-NEXT: }
  cudaDeviceProp deviceProp;
  sprintf(devstr, "pci %x:%x:%x", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
  if (deviceProp.concurrentKernels) {
  }
  if (!deviceProp.canMapHostMemory){
  }
}

void fun4() {
  // CHECK: std::vector<dpct::queue_ptr> vec1;
  // CHECK-NEXT: vec1.push_back(dpct::queue_ptr());
  // CHECK-NEXT: std::vector<dpct::queue_ptr> vec2;
  // CHECK-NEXT: vec2.push_back(dpct::queue_ptr());
  // CHECK-NEXT: dpct::queue_ptr a1 = dpct::queue_ptr();
  // CHECK-NEXT: dpct::queue_ptr a2 = dpct::queue_ptr();
  std::vector<cudaStream_t> vec1;
  vec1.push_back(cudaStream_t());
  std::vector<CUstream> vec2;
  vec2.push_back(CUstream());
  cudaStream_t a1 = cudaStream_t();
  CUstream a2 = CUstream();
}

namespace {
// CHECK: dpct::memcpy_direction K;
// CHECK-NEXT: dpct::memcpy_direction fun(dpct::memcpy_direction);
cudaMemcpyKind K;
cudaMemcpyKind fun(cudaMemcpyKind);
}


namespace {
// CHECK: int M;
// CHECK-NEXT: int fun(int);
cudaComputeMode M;
cudaComputeMode fun(cudaComputeMode);
}

// CHECK: void foo_2(dpct::library_data_t a1, dpct::library_data_t a2, dpct::library_data_t a3, dpct::library_data_t a4) {
// CHECK-NEXT:   dpct::library_data_t b1 = a1;
// CHECK-NEXT:   dpct::library_data_t b2 = a2;
// CHECK-NEXT:   dpct::library_data_t b3 = a3;
// CHECK-NEXT:   dpct::library_data_t b4 = a4;
// CHECK-NEXT: }
void foo_2(cudaDataType_t a1, cudaDataType a2, cublasDataType_t a3, cublasComputeType_t a4) {
  cudaDataType_t b1 = a1;
  cudaDataType b2 = a2;
  cublasDataType_t b3 = a3;
  cublasComputeType_t b4 = a4;
}

__device__ void foo_3() {
  // CHECK: sycl::range<3> d3 = {3, 2, 1}, *pd3 = &d3;
  dim3 d3 = {1, 2, 3}, *pd3 = &d3;
  int64_t m = 0;
  // CHECK: m = sycl::min(m, int64_t((*pd3)[2]));
  // CHECK-NEXT: m = sycl::min(m, int64_t((*pd3)[1]));
  // CHECK-NEXT: m = sycl::min(m, int64_t((*pd3)[0]));
  // CHECK-NEXT: m = sycl::min(m, int64_t(d3[2]));
  // CHECK-NEXT: m = sycl::min(m, int64_t(d3[1]));
  // CHECK-NEXT: m = sycl::min(m, int64_t(d3[0]));
  m = std::min(m, int64_t{pd3->x});
  m = std::min(m, int64_t{pd3->y});
  m = std::min(m, int64_t{pd3->z});
  m = std::min(m, int64_t{d3.x});
  m = std::min(m, int64_t{d3.y});
  m = std::min(m, int64_t{d3.z});
}

template <typename integer>
constexpr inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

void foo_4() {
  const int64_t num_irows = 32;
  const int64_t num_orows = 32;
  // CHECK: sycl::range<3> threads(1, 1, 32);
  dim3 threads(32);
  int64_t maxGridDim = 1024;
  // CHECK: sycl::range<3> grid_1(1, std::min(maxGridDim, ceil_div(num_irows, int64_t(threads[2]))), std::min(maxGridDim, num_orows));
  dim3 grid_1(std::min(maxGridDim, num_orows), std::min(maxGridDim, ceil_div(num_irows, int64_t{threads.x})));

  int row_size = 16;
  // CHECK: sycl::range<3> grid_2(1, 1, std::min<int>(maxGridDim, ceil_div(row_size, int(threads[1]))));
  dim3 grid_2(std::min<int>(maxGridDim, ceil_div(row_size, int(threads.y))));

  // CHECK: int64_t m = int64_t(threads[1]);
  int64_t m = int64_t{threads.y};
  // CHECK: m = long(threads[1]);
  m = long{threads.y};
  using INT64 = int64_t;
  // CHECK: m = std::min(long(threads[2]), INT64(threads[0]));
  m = std::min(long{threads.x}, INT64{threads.z});

  int num = 1024;
  // CHECK: m = long{num};
  m = long{num};
  // CHECK: m = std::min(long(threads[2]), INT64{num});
  m = std::min(long{threads.x}, INT64{num});

  struct CFoo {
    int64_t a = 0;
    CFoo(int64_t b) : a(b) {}
    operator int64_t() { return a; }
  };
  // CHECK: CFoo cfoo{num};
  CFoo cfoo{num};
  // CHECK: m = std::min(long(threads[2]), int64_t{cfoo});
  m = std::min(long{threads.x}, int64_t{cfoo});
}