// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: cat %s > %T/macro_test.cu
// RUN: cat %S/macro_test.h > %T/macro_test.h
// RUN: cd %T
// RUN: rm -rf %T/macro_test_output
// RUN: mkdir %T/macro_test_output
// RUN: dpct -out-root %T/macro_test_output macro_test.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/macro_test_output/macro_test.dp.cpp --match-full-lines macro_test.cu
// RUN: FileCheck --input-file %T/macro_test_output/macro_test.h --match-full-lines macro_test.h
#include "cuda.h"
#include <math.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cusolverDn.h>
#include <stdexcept>

#include <stdio.h>

// CHECK: #include <algorithm>

#include "macro_test.h"

#include <cublas_v2.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

#define CUDA_NUM_THREADS 1024+32
#define GET_BLOCKS(n,t)  1+n+t-1
#define GET_BLOCKS2(n,t) 1+n+t
#define GET_BLOCKS3(n,t) n+t-1
#define GET_BLOCKS4(n,t) n+t

#define NESTMACRO(k) k
#define NESTMACRO2(k) NESTMACRO(k)
#define NESTMACRO3(k) NESTMACRO2(k)

class DDD{
public:
  dim3* A;
  dim3 B;
};
#define CALL(x) x;

#define EMPTY_MACRO(x) x
//CHECK:#define GET_MEMBER_MACRO(x) x[1] = 5
#define GET_MEMBER_MACRO(x) x.y = 5

__global__ void foo_kernel() {}

//CHECK: void foo_kernel2(int a, int b
//CHECK-NEXT:   #ifdef MACRO_CC
//CHECK-NEXT:   , int c
//CHECK-NEXT:   #endif
//CHECK-NEXT:   , const sycl::nd_item<3> &item_ct1) {
//CHECK-NEXT:     int x = item_ct1.get_group(2);
//CHECK-NEXT:   }
__global__ void foo_kernel2(int a, int b
#ifdef MACRO_CC
, int c
#endif
) {
  int x = blockIdx.x;
}

__global__ void foo2(){
  // CHECK: #define IMUL(a, b) sycl::mul24(a, b)
  // CHECK-NEXT: int vectorBase = IMUL(1, 2);
  #define IMUL(a, b) __mul24(a, b)
  int vectorBase = IMUL(1, 2);
}

__global__ void foo3(int x, int y) {}

void foo() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  DDD d3;

// CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
#ifdef __CUDA_ARCH__
  // CHECK: int CA = DPCT_COMPATIBILITY_TEMP;
  int CA = __CUDA_ARCH__;
#endif


  // CHECK: (*d3.A)[2] = 3;
  // CHECK-NEXT: d3.B[2] = 2;
  // CHECK-NEXT: EMPTY_MACRO(d3.B[2]);
  // CHECK-NEXT: GET_MEMBER_MACRO(d3.B);
  d3.A->x = 3;
  d3.B.x = 2;
  EMPTY_MACRO(d3.B.x);
  GET_MEMBER_MACRO(d3.B);

  int outputThreadCount = 512;

  //CHECK: /*
  //CHECK-NEXT: DPCT1038:{{[0-9]+}}: When the kernel function name is used as a macro argument, the
  //CHECK-NEXT: migration result may be incorrect. You need to verify the definition of the
  //CHECK-NEXT: macro.
  //CHECK-NEXT: */
  //CHECK-NEXT: CALL(([&]() {
  //CHECK-NEXT:   q_ct1.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           foo_kernel();
  //CHECK-NEXT:         });
  //CHECK-NEXT: }()))
  CALL( (foo_kernel<<<1, 2, 0>>>()) )

  //CHECK: #define AA 3
  //CHECK-NEXT: #define MCALL                                                                  \
  //CHECK-NEXT: q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 2) *               \
  //CHECK-NEXT:                                          sycl::range<3>(1, 1, 2 * AA),       \
  //CHECK-NEXT:                                      sycl::range<3>(1, 1, 2 * AA)),          \
  //CHECK-NEXT:                    [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  //CHECK-NEXT: MCALL
  #define AA 3
  #define MCALL foo_kernel<<<dim3(2,1), 2*AA, 0>>>();
  MCALL

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS(outputThreadCount, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel();
  // CHECK-NEXT:       });
  foo_kernel<<<GET_BLOCKS(outputThreadCount, outputThreadCount), 2, 0>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel();
  // CHECK-NEXT:       });
  foo_kernel<<<GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS), 0, 0>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel();
  // CHECK-NEXT:       });
  foo_kernel<<<GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount), 0, 0>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS4(outputThreadCount, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel();
  // CHECK-NEXT:       });
  foo_kernel<<<GET_BLOCKS4(outputThreadCount, CUDA_NUM_THREADS), 2, 0>>>();

  // Test if SIGABRT.
  // No check here because the generated code needs further fine tune.
  #define MACRO_CALL(a, b) foo_kernel<<<a, b, 0>>>();
  MACRO_CALL(0,0)

// CHECK: #define HANDLE_GPU_ERROR(err) \
// CHECK-NEXT: do \
// CHECK-NEXT: { \
// CHECK-NEXT:     if (err != 0) \
// CHECK-NEXT:     { \
// CHECK-NEXT:         int currentDevice; \
// CHECK-NEXT:         currentDevice = dpct::dev_mgr::instance().current_device_id(); \
// CHECK-NEXT:     } \
// CHECK-NEXT: } while (0)
#define HANDLE_GPU_ERROR(err) \
do \
{ \
    if(err != cudaSuccess) \
    { \
        int currentDevice; \
        cudaGetDevice(&currentDevice); \
    } \
} \
while(0)

HANDLE_GPU_ERROR(0);

// CHECK: #define cbrt(x) pow((double)x, (double)(1.0 / 3.0))
// CHECK-NEXT: double DD = sqrt(cbrt(5.9)) / sqrt(cbrt(3.2));
#define cbrt(x) pow((double)x,(double)(1.0/3.0))
  double DD = sqrt(cbrt(5.9)) / sqrt(cbrt(3.2));

// CHECK: #define NNBI(x) floor(x+0.5)
// CHECK-NEXT: NNBI(3.0);
#define NNBI(x) floor(x+0.5)
NNBI(3.0);

// CHECK: #define PI acos(-1)
#define PI acos(-1)
// CHECK: double cosine = cos(2 * PI);
double cosine = cos(2 * PI);

//CHECK: #define MACRO_KC                                                                    \
//CHECK-NEXT:   q_ct1.parallel_for(                                                          \
//CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2),   \
//CHECK-NEXT:                           sycl::range<3>(1, 1, 2)),                            \
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
#define MACRO_KC foo_kernel<<<2, 2, 0>>>();

//CHECK: MACRO_KC
MACRO_KC

//CHECK: #define HARD_KC(NAME, a, b, c, d)                                              \
//CHECK-NEXT:   q_ct1.submit([&](sycl::handler &cgh) {                                       \
//CHECK-NEXT:     int c_ct0 = c;                                                            \
//CHECK-NEXT:     int d_ct1 = d;                                                            \
//CHECK:     cgh.parallel_for(                                                          \
//CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, a) * sycl::range<3>(1, 1, b),   \
//CHECK-NEXT:                           sycl::range<3>(1, 1, b)),                            \
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) { foo3(c_ct0, d_ct1); });               \
//CHECK-NEXT:   });
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1038:{{[0-9]+}}: When the kernel function name is used as a macro argument, the
//CHECK-NEXT: migration result may be incorrect. You need to verify the definition of the
//CHECK-NEXT: macro.
//CHECK-NEXT: */
//CHECK-NEXT: HARD_KC(foo3, 3, 2, 1, 0)
#define HARD_KC(NAME,a,b,c,d) NAME<<<a,b,0>>>(c,d);
HARD_KC(foo3,3,2,1,0)

//CHECK: #define MACRO_KC2(a, b, c, d)                                                       \
//CHECK-NEXT:   q_ct1.submit([&](sycl::handler &cgh) {                                       \
//CHECK-NEXT:     int c_ct0 = c;                                                            \
//CHECK-NEXT:     int d_ct1 = d;                                                            \
//CHECK-NEXT:                                                                                \
//CHECK-NEXT:     cgh.parallel_for(sycl::nd_range<3>(a * b, b),                  \
//CHECK-NEXT:                      [=](sycl::nd_item<3> item_ct1) { foo3(c_ct0, d_ct1); });  \
//CHECK-NEXT:   });
#define MACRO_KC2(a,b,c,d) foo3<<<a, b, 0>>>(c,d);

dim3 griddim = 2;
dim3 threaddim = 32;

// CHECK: MACRO_KC2(griddim,threaddim,1,0)
MACRO_KC2(griddim,threaddim,1,0)

// CHECK: MACRO_KC2(3,2,1,0)
MACRO_KC2(3,2,1,0)

// CHECK: MACRO_KC2(sycl::range<3>(5, 4, 3), 2, 1, 0)
MACRO_KC2(dim3(5,4,3),2,1,0)

int *a;
//CHECK: NESTMACRO3(a = (int *)sycl::malloc_device(100, q_ct1));
NESTMACRO3(cudaMalloc(&a,100));

//test if parse error, no check
int b;
#if ( __CUDACC_VER_MAJOR__ >= 8 ) && (__CUDA_ARCH__ >= 600 )
  // DPCT should visit this path
#else
  // If DPCT visit this path, b is redeclared.
  int b;
#endif

  //CHECK: q_ct1.parallel_for(
  //CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2),
  //CHECK-NEXT:                         sycl::range<3>(1, 1, 2)),
  //CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:         foo_kernel2(3, 3, item_ct1);
  //CHECK-NEXT:       });
  foo_kernel2<<<2, 2, 0>>>(3,3
    #ifdef MACRO_CC
    , 2
    #endif
  );

  #define SIZE3    (100*1024*1024)
  unsigned char *dev_buffer;
  unsigned char *buffer = (unsigned char*)malloc(500);
  //CHECK: q_ct1.memcpy(dev_buffer, buffer, SIZE3).wait();
  cudaMemcpy( dev_buffer, buffer, SIZE3, cudaMemcpyHostToDevice);
}

// CHECK: template <class T>
// CHECK-NEXT: bool reallocate_host(T **pp, int *curlen, const int newlen,
// CHECK-NEXT:                      /*
// CHECK-NEXT:                      DPCT1048:{{[0-9]+}}: The original value cudaHostAllocDefault is not
// CHECK-NEXT:                      meaningful in the migrated code and was removed or replaced
// CHECK-NEXT:                      with 0. You may need to check the migrated code.
// CHECK-NEXT:                      */
// CHECK-NEXT:                      const float fac = 1.0f, const unsigned int flag = 0) {
// CHECK-NEXT:   return true;//reallocate_host_T((void **)pp, curlen, newlen, fac, flag, sizeof(T));
// CHECK-NEXT: }
template <class T>
  bool reallocate_host(T **pp, int *curlen, const int newlen,
                       const float fac=1.0f, const unsigned int flag = cudaHostAllocDefault) {
  return true;//reallocate_host_T((void **)pp, curlen, newlen, fac, flag, sizeof(T));
}

bool fooo() {
  int *force_ready_queue;
  int force_ready_queue_size;
  int npatches;
  // CHECK: return reallocate_host<int>(
  // CHECK-NEXT:     &force_ready_queue, &force_ready_queue_size,
  // CHECK-NEXT:     /*
  // CHECK-NEXT:     DPCT1048:{{[0-9]+}}: The original value cudaHostAllocMapped is not meaningful in
  // CHECK-NEXT:     the migrated code and was removed or replaced with 0. You may need to
  // CHECK-NEXT:     check the migrated code.
  // CHECK-NEXT:     */
  // CHECK-NEXT:     npatches, 1.2f, 0);
  return reallocate_host<int>(&force_ready_queue, &force_ready_queue_size,
                              npatches, 1.2f, cudaHostAllocMapped);
}

void bar() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1048:{{[0-9]+}}: The original value cudaHostAllocDefault is not meaningful in the
  // CHECK-NEXT: migrated code and was removed or replaced with 0. You may need to check the
  // CHECK-NEXT: migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: int i = 0;
  int i = cudaHostAllocDefault;
  // CHECK: /*
  // CHECK-NEXT: DPCT1048:{{[0-9]+}}: The original value cudaHostAllocMapped is not meaningful in the
  // CHECK-NEXT: migrated code and was removed or replaced with 0. You may need to check the
  // CHECK-NEXT: migrated code.
  // CHECK-NEXT: */
  i = cudaHostAllocMapped;
  // CHECK: /*
  // CHECK-NEXT: DPCT1048:{{[0-9]+}}: The original value cudaHostAllocPortable is not meaningful in the
  // CHECK-NEXT: migrated code and was removed or replaced with 0. You may need to check the
  // CHECK-NEXT: migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: i = 0;
  i = cudaHostAllocPortable;
  // CHECK: /*
  // CHECK-NEXT: DPCT1048:{{[0-9]+}}: The original value cudaHostAllocWriteCombined is not meaningful in
  // CHECK-NEXT: the migrated code and was removed or replaced with 0. You may need to check
  // CHECK-NEXT: the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: i = 0;
  i = cudaHostAllocWriteCombined;
}
// CHECK: #define BB b
// CHECK-NEXT: #define AAA int *a
// CHECK-NEXT: #define BBB int *BB
#define BB b
#define AAA int *a
#define BBB int *BB

// CHECK: #define CCC AAA, float *sp_lj, float *sp_coul, int *ljd, sycl::local_accessor<double, 2> la, int *b=0
// CHECK-NEXT: #define CC AAA, BBB
#define CCC AAA, int *b=0
#define CC AAA, BBB

// CHECK: #define CCCC(x) void fooc(x)
// CHECK-NEXT: #define CCCCC(x) void foocc(x, float *sp_lj, float *sp_coul, int *ljd, sycl::local_accessor<double, 2> la)
#define CCCC(x) __device__ void fooc(x)
#define CCCCC(x) __device__ void foocc(x)

// CHECK: #define XX(x) void foox(x, float *sp_lj, float *sp_coul, int *ljd, sycl::local_accessor<double, 2> la)
// CHECK-NEXT: #define FF XX(CC)
#define XX(x) __device__ void foox(x)
#define FF XX(CC)

// CHECK: FF
// CHECK-NEXT: {
// CHECK-NEXT: }
FF
{
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[1];
  __shared__ double la[8][1];
}

// CHECK: CCCCC(int *a)
// CHECK-NEXT: {
// CHECK-NEXT: }
CCCCC(int *a)
{
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[1];
  __shared__ double la[8][1];
}


// CHECK: CCCC(CCC)
// CHECK-NEXT: {
// CHECK-NEXT: }
CCCC(CCC)
{
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[1];
  __shared__ double la[8][1];
}

// CHECK: #define FFF void foo(AAA, BBB, float *sp_lj, float *sp_coul, int *ljd, sycl::local_accessor<double, 2> la)
#define FFF __device__ void foo(AAA, BBB)

// CHECK: FFF
// CHECK-NEXT: {
// CHECK-NEXT: }
FFF
{
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[1];
  __shared__ double la[8][1];

}

// CHECK: #define FFFFF(aaa,bbb) void foo4(const int * __restrict__ aaa, const float * __restrict__ bbb, int *c, BBB, const sycl::nd_item<3> &item_ct1, float *sp_lj, float *sp_coul, int *ljd, sycl::local_accessor<double, 2> la)
#define FFFFF(aaa,bbb) __device__ void foo4(const int * __restrict__ aaa, const float * __restrict__ bbb, int *c, BBB)

// CHECK: FFFFF(pos, q)
// CHECK-NEXT: {
// CHECK-EMPTY:
// CHECK-NEXT: const int tid = item_ct1.get_local_id(2);
// CHECK-NEXT: }
FFFFF(pos, q)
{
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[1];
  __shared__ double la[8][1];
  const int tid = threadIdx.x;
}

// CHECK: #define FFFFFF(aaa,bbb) void foo5(const int * __restrict__ aaa, const float * __restrict__ bbb, const sycl::nd_item<3> &item_ct1, float *sp_lj, float *sp_coul, int *ljd, sycl::local_accessor<double, 2> la)
#define FFFFFF(aaa,bbb) __device__ void foo5(const int * __restrict__ aaa, const float * __restrict__ bbb)

// CHECK: FFFFFF(pos, q)
// CHECK-NEXT: {
// CHECK-EMPTY:
// CHECK-NEXT: const int tid = item_ct1.get_local_id(2);
// CHECK-NEXT: }
FFFFFF(pos, q)
{
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[1];
  __shared__ double la[8][1];
  const int tid = threadIdx.x;
}

// CHECK: void foo6(AAA, BBB, float *sp_lj, float *sp_coul, int *ljd,
// CHECK-NEXT:   sycl::local_accessor<double, 2> la)
// CHECK-NEXT: {
// CHECK-NEXT: }
__device__ void foo6(AAA, BBB)
{
   __shared__ float sp_lj[4];
   __shared__ float sp_coul[4];
   __shared__ int ljd[1];
   __shared__ double la[8][1];
}


//CHECK: #define MM __umul24
//CHECK-NEXT: #define MUL(a, b) sycl::mul24((unsigned int)a, (unsigned int)b)
//CHECK-NEXT: void foo7(const sycl::nd_item<3> &item_ct1) {
//CHECK-NEXT:   unsigned int tid = MUL(item_ct1.get_local_range(2), item_ct1.get_group(2)) +
//CHECK-NEXT:       item_ct1.get_local_range(2);
//CHECK-NEXT:   unsigned int tid2 = sycl::mul24((unsigned int)item_ct1.get_local_range(2),
//CHECK-NEXT:                                   (unsigned int)item_ct1.get_group_range(2));
//CHECK-NEXT: }
#define MM __umul24
#define MUL(a, b) __umul24(a, b)
__global__ void foo7() {
  unsigned int      tid = MUL(blockDim.x, blockIdx.x) + blockDim.x;
  unsigned int      tid2 = MM(blockDim.x, gridDim.x);
}


//CHECK: void foo8(){
//CHECK-NEXT:   #define SLOW(X) X
//CHECK-NEXT:   double* data;
//CHECK-NEXT:   unsigned long long int tid;
//CHECK-NEXT:   SLOW(dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
//CHECK-NEXT:            &data[1], tid);
//CHECK-NEXT:        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
//CHECK-NEXT:            &data[1], tid + 1);
//CHECK-NEXT:        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
//CHECK-NEXT:            &data[2], tid + 2););
//CHECK-NEXT: }
__global__ void foo8(){
#define SLOW(X) X
  double* data;
  unsigned long long int tid;
  SLOW(atomicAdd(&data[1], tid);
  atomicAdd(&data[1], tid + 1);
  atomicAdd(&data[2], tid + 2););
}

//CHECK: #define DFABS(x) (double)sycl::fabs((x))
//CHECK-NEXT: #define MAX(x, y) dpct::max(x, y)
//CHECK-NEXT: void foo9(){
//CHECK-NEXT:   double a,b,c;
//CHECK-NEXT:   MAX(a, sycl::sqrt(DFABS(b)));
//CHECK-NEXT: }
#define DFABS(x) (double) fabs((x))
#define MAX(x, y) max(x, y)
__global__ void foo9(){
  double a,b,c;
  MAX(a, sqrt(DFABS(b)));
}



//CHECK: #define My_PI  3.14159265358979
//CHECK-NEXT: #define g2r(x)  (((double)(x))*My_PI/180)
//CHECK-NEXT: #define sindeg(x) sin(g2r(x))
//CHECK-NEXT: void foo10()
//CHECK-NEXT: {
//CHECK-NEXT:   sindeg(5);
//CHECK-NEXT: }
#define My_PI  3.14159265358979
#define g2r(x)  (((double)(x))*My_PI/180)
#define sindeg(x) sin(g2r(x))
void foo10()
{
  sindeg(5);
}


template<int a, int b>
__global__ void templatefoo(){
  int x = a;
  int y = b;
}
//CHECK: #define AAA 15 + 3
//CHECK-NEXT: #define CCC <<<1,1>>>()
//CHECK-NEXT: #define KERNEL(A, B)                                                           \
//CHECK-NEXT:   dpct::get_in_order_queue().parallel_for(                                      \
//CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),   \
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) { templatefoo<A, B>(); });
//CHECK-NEXT: #define CALL_KERNEL(C, D) KERNEL(C, D); int a = 0;
//CHECK-NEXT: #define CALL_KERNEL2(E, F) CALL_KERNEL(E, F)
//CHECK-NEXT: void templatefoo2(){
//CHECK-NEXT:   CALL_KERNEL2(8, AAA)
//CHECK-NEXT: }
#define AAA 15 + 3
#define CCC <<<1,1>>>()
#define KERNEL(A, B) templatefoo<A,B>CCC
#define CALL_KERNEL(C, D) KERNEL(C, D); int a = 0;
#define CALL_KERNEL2(E, F) CALL_KERNEL(E, F)
void templatefoo2(){
  CALL_KERNEL2(8, AAA)
}

//CHECK: void foo11(const sycl::nd_item<3> &item_ct1){
//CHECK-NEXT:   sycl::exp((double)(THREAD_IDX_X));
//CHECK-NEXT: }
__global__ void foo11(){
  exp(THREAD_IDX_X);
}

//CHECK: /*
//CHECK-NEXT: DPCT1055:{{[0-9]+}}: Vector types with size 1 are migrated to the corresponding
//CHECK-NEXT: fundamental types, which cannot be inherited. You need to rewrite the code.
//CHECK-NEXT: */
//CHECK-NEXT: #define VECTOR_TYPE_DEF(type)                                                  \
//CHECK-NEXT:                                                                                \
//CHECK-NEXT:   struct MyVector : type                                                       \
//CHECK-NEXT:   {                                                                            \
//CHECK-NEXT:     typedef type Type;                                                         \
//CHECK-NEXT:     __dpct_inline__ MyVector operator+(const MyVector &other) const {          \
//CHECK-NEXT:       MyVector retval;                                                         \
//CHECK-NEXT:       retval = *this + other;                                                  \
//CHECK-NEXT:       return retval;                                                           \
//CHECK-NEXT:     }                                                                          \
//CHECK-NEXT:     __dpct_inline__ MyVector operator-(const MyVector &other) const {          \
//CHECK-NEXT:       MyVector retval;                                                         \
//CHECK-NEXT:       retval = *this - other;                                                  \
//CHECK-NEXT:       return retval;                                                           \
//CHECK-NEXT:     }                                                                          \
//CHECK-NEXT:   };                                                                           \
//CHECK-NEXT:                                                                                \
//CHECK-NEXT:   struct MyVector2 : sycl::type##2                                             \
//CHECK-NEXT:   {                                                                            \
//CHECK-NEXT:     typedef sycl::type##2 Type;                                                \
//CHECK-NEXT:     __dpct_inline__ MyVector2 operator+(const MyVector2 &other) const {        \
//CHECK-NEXT:       MyVector2 retval;                                                        \
//CHECK-NEXT:       retval.x() = x() + other.x();                                            \
//CHECK-NEXT:       retval.y() = y() + other.y();                                            \
//CHECK-NEXT:       return retval;                                                           \
//CHECK-NEXT:     }                                                                          \
//CHECK-NEXT:     __dpct_inline__ MyVector2 operator-(const MyVector2 &other) const {        \
//CHECK-NEXT:       MyVector2 retval;                                                        \
//CHECK-NEXT:       retval.x() = x() - other.x();                                            \
//CHECK-NEXT:       retval.y() = y() - other.y();                                            \
//CHECK-NEXT:       return retval;                                                           \
//CHECK-NEXT:     }                                                                          \
//CHECK-NEXT:   };

#define VECTOR_TYPE_DEF(type)                                                                           \
                                                                                                        \
    struct MyVector : type##1                                                                           \
    {                                                                                                   \
        typedef type##1   Type;                                                                         \
        __host__ __device__ __forceinline__ MyVector operator+(const MyVector &other) const {           \
        MyVector retval;                                                                                \
            retval.x = x + other.x;                                                                     \
            return retval;                                                                              \
        }                                                                                               \
        __host__ __device__ __forceinline__ MyVector operator-(const MyVector &other) const {           \
        MyVector retval;                                                                                \
            retval.x = x - other.x;                                                                     \
            return retval;                                                                              \
        }                                                                                               \
    };                                                                                                  \
                                                                                                        \
    struct MyVector2 : type##2                                                                          \
    {                                                                                                   \
        typedef type##2 Type;                                                                           \
        __host__ __device__ __forceinline__ MyVector2 operator+(const MyVector2 &other) const {         \
            MyVector2 retval;                                                                           \
            retval.x = x + other.x;                                                                     \
            retval.y = y + other.y;                                                                     \
            return retval;                                                                              \
        }                                                                                               \
        __host__ __device__ __forceinline__ MyVector2 operator-(const MyVector2 &other) const {         \
            MyVector2 retval;                                                                           \
            retval.x = x - other.x;                                                                     \
            retval.y = y - other.y;                                                                     \
            return retval;                                                                              \
        }                                                                                               \
    };

VECTOR_TYPE_DEF(int)

//CHECK: typedef float real;
//CHECK-NEXT: #define POW(x, y) dpct::pow(x, y)
//CHECK-NEXT: #define POW2(x, y) x *x
//CHECK-NEXT: /*
//CHECK-NEXT: DPCT1064:{{[0-9]+}}: Migrated pow call is used in a macro/template definition and may
//CHECK-NEXT: not be valid for all macro/template uses. Adjust the code.
//CHECK-NEXT: */
//CHECK-NEXT: #define POW3(x, y) dpct::pow(x, y)
//CHECK: #define SQRT(x) sycl::sqrt(x)
//CHECK-NEXT: void foo12(){
//CHECK-NEXT: real *vx;
//CHECK-NEXT: real *vy;
//CHECK-NEXT: int id;
//CHECK-NEXT: real v2 = SQRT(SQRT(POW(vx[id], 2.0) + POW(vy[id], 2.0)));
//CHECK-NEXT: real v3 = POW2(vx[id], 2);
//CHECK-NEXT: real v4 = POW3(vx[id], 3.0);
//CHECK-NEXT: real v5 = POW3(vx[id], 2);
//CHECK-NEXT: }
typedef float real;
#define POW(x,y)    powf(x,y)
#define POW2(x,y)    pow(x,y)
#define POW3(x,y)    pow(x,y)
#define SQRT(x)     sqrtf(x)
__global__ void foo12(){
real *vx;
real *vy;
int id;
real v2 = SQRT(SQRT(POW(vx[id], 2.0) + POW(vy[id], 2.0)));
real v3 = POW2(vx[id], 2);
real v4 = POW3(vx[id], 3.0);
real v5 = POW3(vx[id], 2);
}

//CHECK: #define CALL(call) call;
//CHECK-NEXT: #define SIZE2 8
//CHECK-NEXT: void foo13(){
//CHECK-NEXT:   int *a;
//CHECK-NEXT:   CALL(a = sycl::malloc_device<int>(SIZE2 * 10, dpct::get_in_order_queue()));
//CHECK-NEXT: }
#define CALL(call) call;
#define SIZE2 8
void foo13(){
  int *a;
  CALL(cudaMalloc(&a, SIZE2 * 10 * sizeof(int)));
}

//CHECK: #define CONST const
//CHECK-NEXT: #define INT2 sycl::int2
//CHECK-NEXT: #define PTR *
//CHECK-NEXT: #define PTR2 PTR
//CHECK-NEXT: #define ALL const sycl::int2 *
//CHECK-NEXT: #define TYPE_PTR(T) T *
//CHECK-NEXT: #define ALL2(C, T, P) C T P
//CHECK-NEXT: #define ALL3(X) X
#define CONST const
#define INT2 int2
#define PTR *
#define PTR2 PTR
#define ALL const int2 *
#define TYPE_PTR(T) T *
#define ALL2(C, T, P) C T P
#define ALL3(X) X

//CHECK: int foo14(){
//CHECK-NEXT:   const sycl::int2 *aaa;
//CHECK-NEXT:   CONST sycl::int3 *bbb;
//CHECK-NEXT:   ALL3(const sycl::int2 *) ccc;
//CHECK-NEXT:   ALL2(const, sycl::int2, *) ddd;
//CHECK-NEXT:   ALL3(const) ALL3(sycl::int2) ALL3(*) eee;
//CHECK-NEXT:   ALL fff;
//CHECK-NEXT:   CONST INT2 PTR ggg;
//CHECK-NEXT:   CONST INT2 PTR2 hhh;
//CHECK-NEXT:   CONST sycl::int3 PTR2 iii;
//CHECK-NEXT:   TYPE_PTR(sycl::int2) jjj;
//CHECK-NEXT:   ALL3(ALL3(const sycl::int2 *)) kkk;
//CHECK-NEXT:   ALL2(const, ALL3(sycl::int2), *) lll;
//CHECK-NEXT: }
int foo14(){
  const int2 *aaa;
  CONST int3 *bbb;
  ALL3(const int2 *) ccc;
  ALL2(const, int2, *) ddd;
  ALL3(const) ALL3(int2) ALL3(*) eee;
  ALL fff;
  CONST INT2 PTR ggg;
  CONST INT2 PTR2 hhh;
  CONST int3 PTR2 iii;
  TYPE_PTR(int2) jjj;
  ALL3(ALL3(const int2 *)) kkk;
  ALL2(const, ALL3(int2), *) lll;
}

//CHECK: #define FABS(a) (sycl::fabs((float)((a).x())) + sycl::fabs((float)((a).y())))
//CHECK-NEXT: static inline double foo16(const sycl::float2 &x) { return FABS(x); }
#define FABS(a)       (fabs((a).x) + fabs((a).y))
__host__ __device__ static inline double foo16(const float2 &x) { return FABS(x); }

//CHECK: #define _mulhilo_(W, Word, NAME)                                               \
//CHECK-NEXT: Word mulhilo##W(Word a, Word b, Word *hip) {                                 \
//CHECK-NEXT:     *hip = NAME(a, b);                                                         \
//CHECK-NEXT:     return a * b;                                                              \
//CHECK-NEXT: }
//CHECK-NEXT: _mulhilo_(64, uint64_t, sycl::mul_hi)
#include "cuda_fp16.h"
#define _mulhilo_(W, Word, NAME)                       \
__device__ Word mulhilo##W(Word a, Word b, Word* hip) { \
    *hip = NAME(a, b);                                 \
    return a*b;                                        \
}
_mulhilo_(64, uint64_t, __umul64hi)




//CHECK: #define AAA __heq
//CHECK-NEXT: #define CALL(x) x
//CHECK-NEXT: #define CALL2(x) CALL(x)
//CHECK-NEXT: #define III CALL(CALL(CALL(h == h_1)))
//CHECK-NEXT: #define JJJ CALL(CALL(CALL(III)))
//CHECK-NEXT: #define KKK JJJ
//CHECK-NEXT: void foo16() {
//CHECK-NEXT:     sycl::half h, h_1, h_2;
//CHECK-NEXT:     sycl::half2 h2, h2_1, h2_2;
//CHECK-NEXT:     bool b;
//CHECK-NEXT:     CALL(CALL(CALL(JJJ)));
//CHECK-NEXT: }
#define AAA __heq
#define CALL(x) x
#define CALL2(x) CALL(x)
#define III CALL(CALL(CALL(AAA (h, h_1))))
#define JJJ CALL(CALL(CALL(III)))
#define KKK JJJ
__global__ void foo16() {
    __half h, h_1, h_2;
    __half2 h2, h2_1, h2_2;
    bool b;
    CALL(CALL(CALL(JJJ)));
}

// [Todo] Macro issue here will fix in issue jira
void foo17(){
  size_t result1, result2;
  int size = 32;
  float* f_A;
  // Error CALL() will be removed
  CALL(CUDA_MEMCPY3D cpy2);
  CUdeviceptr f_D = 0;
  CALL(cuMemAlloc(&f_D, size));
}

//CHECK: #define CONCATE(name) cuda##name
//CHECK-NEXT: typedef dpct::queue_ptr stream_t2;
//CHECK-NEXT: typedef dpct::event_ptr event_t2;
#define CONCATE(name) cuda##name
typedef CONCATE(Stream_t) stream_t2;
typedef CONCATE(Event_t) event_t2;

//CHECK: void foo18() {
//CHECK-NEXT:   dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK-NEXT:   dpct::event_ptr event;
//CHECK-NEXT:   event->wait_and_throw();
//CHECK-NEXT:   stream_t2 *stream;
//CHECK-NEXT:   stream_t2 stream2;
//CHECK-NEXT:   *(stream) = dev_ct1.create_queue();
//CHECK-NEXT:   unsigned int flags;
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
//CHECK-NEXT:   */
//CHECK-NEXT:   *(stream) = dev_ct1.create_queue();
//CHECK-NEXT:   int priority;
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag and priority options.
//CHECK-NEXT:   */
//CHECK-NEXT:   *(stream) = dev_ct1.create_queue();
//CHECK-NEXT:   dev_ct1.destroy_queue(stream2);
//CHECK-NEXT: }
void foo18(){
  cudaEvent_t event;
  CONCATE(EventSynchronize)(event);
  stream_t2 *stream;
  stream_t2 stream2;
  CONCATE(StreamCreate)(stream);
  unsigned int flags;
  CONCATE(StreamCreateWithFlags)(stream, flags);
  int priority;
  CONCATE(StreamCreateWithPriority)(stream, flags, priority);
  CONCATE(StreamDestroy)(stream2);
}

// CHECK: static const int streamDefault2 = 0;
// CHECK-NEXT: static const int streamDefault = CALL(0);
// CHECK-NEXT: static const int streamNonBlocking = 0;
// CHECK-NEXT: static const dpct::queue_ptr streamDefault3 = &dpct::get_in_order_queue();
// CHECK-NEXT: static const dpct::queue_ptr streamDefault4 = CALL(&dpct::get_in_order_queue());
static const int streamDefault2 = cudaStreamDefault;
static const int streamDefault = CALL(CONCATE(StreamDefault));
static const int streamNonBlocking = CONCATE(StreamNonBlocking);
static const cudaStream_t streamDefault3 = cudaStreamDefault;
static const cudaStream_t streamDefault4 = CALL(cudaStreamDefault);


//     CHECK:#define CMC_PROFILING_BEGIN()                                                  \
//CHECK-NEXT:  dpct::event_ptr start;                                                         \
//CHECK-NEXT:  std::chrono::time_point<std::chrono::steady_clock> start_ct1;                \
//CHECK-NEXT:  dpct::event_ptr stop;                                                          \
//CHECK-NEXT:  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;                 \
//CHECK-NEXT:  if (CMC_profile)                                                             \
//CHECK-NEXT:  {                                                                            \
//CHECK-NEXT:    start = new sycl::event();                                                 \
//CHECK-NEXT:    stop = new sycl::event();                                                  \
//CHECK-NEXT:    start_ct1 = std::chrono::steady_clock::now();                              \
//CHECK-NEXT:    *start = q_ct1.ext_oneapi_submit_barrier();                                \
//CHECK-NEXT:  }
#define CMC_PROFILING_BEGIN()                                                                                      \
  cudaEvent_t start;                                                                                               \
  cudaEvent_t stop;                                                                                                \
  if (CMC_profile)                                                                                                 \
  {                                                                                                                \
    cudaEventCreate(&start);                                                                                       \
    cudaEventCreate(&stop);                                                                                        \
    cudaGetLastError();                                                                                            \
    cudaEventRecord(start);                                                                                        \
  }


//     CHECK:#define CMC_PROFILING_END(lineno)                                              \
//CHECK-NEXT:  if (CMC_profile)                                                             \
//CHECK-NEXT:  {                                                                            \
//CHECK-NEXT:    stop_ct1 = std::chrono::steady_clock::now();                               \
//CHECK-NEXT:    *stop = q_ct1.ext_oneapi_submit_barrier();                                 \
//CHECK-NEXT:    stop->wait_and_throw();                                                    \
//CHECK-NEXT:    float time = 0.0f;                                                         \
//CHECK-NEXT:    time = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)      \
//CHECK-NEXT:               .count();                                                       \
//CHECK-NEXT:    dpct::destroy_event(start);                                                \
//CHECK-NEXT:    dpct::destroy_event(stop);                                                 \
//CHECK-NEXT:  }                                                                            \
//CHECK-NEXT:  dpct::err0 error = 0;
#define CMC_PROFILING_END(lineno)                                                                          \
  if (CMC_profile)                                                                                         \
  {                                                                                                        \
    cudaEventRecord(stop);                                                                                 \
    cudaEventSynchronize(stop);                                                                            \
    float time = 0.0f;                                                                                     \
    cudaEventElapsedTime(&time, start, stop);                                                              \
    cudaEventDestroy(start);                                                                               \
    cudaEventDestroy(stop);                                                                                \
  }                                                                                                        \
  cudaError_t error = cudaGetLastError();                                                                  \
  if (error)                                                                                               \
  {                                                                                                        \
    printf("%s\nCUDA ERROR!!! Detected at end of CMC_PROFILING_END in BsplineJastrowCudaPBC line %d!!!\n", \
           cudaGetErrorString(error),                                                                      \
           lineno);                                                                                        \
    exit(1);                                                                                               \
  }

void foo20() {
  bool CMC_profile = true;
  CMC_PROFILING_BEGIN();
  CMC_PROFILING_END(__LINE__);
}

//CHECK: #define CALLSHFLSYNC(x)                                                        \
//CHECK-NEXT: dpct::select_from_sub_group(item_ct1.get_sub_group(), x, 3 ^ 1);
#define CALLSHFLSYNC(x) __shfl_sync(0xffffffff, x, 3 ^ 1);
//CHECK: #define CALLANYSYNC(x)                                                         \
//CHECK-NEXT:   sycl::any_of_group(                                                          \
//CHECK-NEXT:       item_ct1.get_sub_group(),                                                \
//CHECK-NEXT:       (0xffffffff &                                                            \
//CHECK-NEXT:        (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&             \
//CHECK-NEXT:           x != 0.0f);
#define CALLANYSYNC(x) __any_sync(0xffffffff, x != 0.0f);

__global__ void foo21(){
  int a;
  CALLSHFLSYNC(a);
  CALLANYSYNC(a);
}


//CHECK: #define FUNCNAME(x) x
//CHECK-NEXT: #define PASS(x) x
//CHECK-NEXT: template <typename T, int X, int Y>
//CHECK-NEXT: void doo(float f, const sycl::stream &stream_ct1) {
//CHECK-NEXT:   stream_ct1 << "doo\n";
//CHECK-NEXT: }
#define FUNCNAME(x) x
#define PASS(x) x
template <typename T, int X, int Y>
__device__ void doo(float f) {
  printf("doo\n");
}

//CHECK: void foo22(const sycl::stream &stream_ct1) {
//CHECK-NEXT:   FUNCNAME(doo)<float, PASS(1 +) 2, SIZE2>(PASS(1 +) 0.0f, stream_ct1);
//CHECK-NEXT: }
__global__ void foo22() {
  FUNCNAME(doo)<float, PASS(1 +) 2, SIZE2>(PASS(1 +) 0.0f);
}

//CHECK: static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("mmx")))
//CHECK-NEXT: foo23(void)
//CHECK-NEXT: {
//CHECK-NEXT:     __builtin_ia32_emms();
//CHECK-NEXT: }
static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("mmx")))
foo23(void)
{
  __builtin_ia32_emms();
}

//CHECK: #define SHFL(x, y, z)                                                          \
//CHECK-NEXT: dpct::select_from_sub_group(item_ct1.get_sub_group(), (x), (y), (z))
#define SHFL(x, y, z) __shfl((x), (y), (z))
__global__ void foo24(){
  int i;
  SHFL(i, i, 16);
}


#include <cublas_v2.h>
int foo25(){
//CHECK: #if defined(MKL_SYCL_HPP)
#if defined(CUBLAS_V2_H_)
#endif

//CHECK: #ifndef MKL_SYCL_HPP
//CHECK-NEXT: #define CUBLAS_V2_H_
#ifndef CUBLAS_V2_H_
#define CUBLAS_V2_H_
float *h_a, *h_b, *h_c;
float *d_C_S;
int a = cublasSetVector(10, sizeof(float), h_a, 11111, d_C_S, 11111);
#endif
return 0;
}

//CHECK:#define AAAAA_Z_MAKE(r, i) sycl::double2(r, i)
//CHECK-NEXT:#define AAAAA_Z_ZERO AAAAA_Z_MAKE(0.0, 0.0)
//CHECK-NEXT:void aaaaa_zprint_vector() {
//CHECK-NEXT:  sycl::double2 z_zero = AAAAA_Z_ZERO;
//CHECK-NEXT:#ifdef COMPLEX
//CHECK-NEXT:#define AAA
//CHECK-NEXT:#else
//CHECK-NEXT:#define BBB
//CHECK-NEXT:#endif
//CHECK-NEXT:}
#define AAAAA_Z_MAKE(r, i) make_cuDoubleComplex(r, i)
#define AAAAA_Z_ZERO AAAAA_Z_MAKE(0.0, 0.0)
void aaaaa_zprint_vector() {
  cuDoubleComplex z_zero = AAAAA_Z_ZERO;
#ifdef COMPLEX
#define AAA
#else
#define BBB
#endif
}

namespace launch_bounds_test {
constexpr uint32_t AAAAA_launch_bounds_test = 1024;
constexpr uint32_t BBBBB_launch_bounds_test = 256;
#define CCCCC_launch_bounds_test(val)          \
  (((val) <= AAAAA_launch_bounds_test) ? (val) \
      : BBBBB_launch_bounds_test)

// CHECK: #define DDDDD_launch_bounds_test(max_threads_per_block) \
// CHECK-NEXT: /*comment*/
// CHECK-NEXT: template <typename T1, typename T2, int I>
#define DDDDD_launch_bounds_test(max_threads_per_block) \
  __launch_bounds__((CCCCC_launch_bounds_test((max_threads_per_block)))) /*comment*/
template <typename T1, typename T2, int I>
DDDDD_launch_bounds_test(512)
__global__ void test() {}

// CHECK: #define EEEEE_launch_bounds_test(max_threads_per_block) \
// CHECK-EMPTY:
// CHECK-NEXT: template <typename T1, typename T2, int I>
#define EEEEE_launch_bounds_test(max_threads_per_block) \
__launch_bounds__((CCCCC_launch_bounds_test((max_threads_per_block))))
template <typename T1, typename T2, int I>
EEEEE_launch_bounds_test(512)
__global__ void test2() {}

#undef CCCCC_launch_bounds_test
#undef DDDDD_launch_bounds_test
#undef EEEEE_launch_bounds_test
}

//     CHECK:#if (defined(DPCT_COMPATIBILITY_TEMP) &&                                       \
//CHECK-NEXT:     !(defined(__clang__) && defined(SYCL_LANGUAGE_VERSION)))
//CHECK-NEXT:__host__ __device__
//CHECK-NEXT:#endif
//CHECK-NEXT:void foo26 () {}
#if (defined(__CUDA_ARCH__) && !(defined(__clang__) && defined(__CUDA__)))
__host__ __device__
#endif
void foo26 () {}

// check not to assert
//CHECK: namespace user_namespace {
//CHECK-NEXT:   template <typename T> struct cufftDoubleComplex {};
//CHECK-NEXT: }
//CHECK-NEXT: #define MACRO_AA(ARG) ARG()
//CHECK-NEXT: template <typename T> void bar() {}
//CHECK-NEXT: #define MACRO_BB() void foo27() { return bar<user_namespace::cufftDoubleComplex<float>>(); }
//CHECK-NEXT: MACRO_AA(MACRO_BB)
namespace user_namespace {
  template <typename T> struct cufftDoubleComplex {};
}
#define MACRO_AA(ARG) ARG()
template <typename T> void bar() {}
#define MACRO_BB() void foo27() { return bar<user_namespace::cufftDoubleComplex<float>>(); }
MACRO_AA(MACRO_BB)


#define CALL_K(...) __VA_ARGS__
void foo28(){
  //CHECK: CALL_K(dpct::get_in_order_queue().parallel_for(
  //CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:     foo_kernel();
  //CHECK-NEXT:   });)
  CALL_K(foo_kernel<<<1,1,0>>>();)
}



#define SIMD_SIZE 32
#define BLOCK_PAIR 256

#define local_allocate_store_charge()                                       \
    __shared__ double red_acc[8][BLOCK_PAIR / SIMD_SIZE];

//CHECK: void foo29(sycl::local_accessor<double, 2> red_acc) {
//CHECK-NEXT: }
__global__ void foo29() {
  local_allocate_store_charge();
}

template<class T1, class T2, int N> __global__ void foo31();

//CHECK: #define FOO31(DIMS)                                                            \
//CHECK-NEXT: q_ct1.parallel_for(                                                          \
//CHECK-NEXT:     sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),     \
//CHECK-NEXT:     [=](sycl::nd_item<3> item_ct1) { foo31<unsigned int, float, DIMS>(); });


#define FOO31(DIMS) foo31<unsigned int, float, DIMS><<<1,1>>>();

//CHECK: {
//CHECK-NEXT:   dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp64});
//CHECK-NEXT:   q_ct1.submit([&](sycl::handler &cgh) {
//CHECK-NEXT:     /*
//CHECK-NEXT:     DPCT1101:{{[0-9]+}}: 'BLOCK_PAIR / SIMD_SIZE' expression was replaced with a
//CHECK-NEXT:     value. Modify the code to use the original expression, provided in
//CHECK-NEXT:     comments, if it is correct.
//CHECK-NEXT:     */
//CHECK-NEXT:     sycl::local_accessor<double, 2> red_acc_acc_ct1(
//CHECK-NEXT:         sycl::range<2>(8, 8 /*BLOCK_PAIR / SIMD_SIZE*/), cgh);

//CHECK:     cgh.parallel_for(
//CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:           foo29(red_acc_acc_ct1);
//CHECK-NEXT:         });
//CHECK-NEXT:   });
//CHECK-NEXT:   }
//CHECK-NEXT:   FOO31(1)
//CHECK-NEXT: }
void foo30(){
  foo29<<<1,1,0>>>();
  FOO31(1)
}



#define VA_CALL2(...) __VA_ARGS__()
#define VA_CALL(...) VA_CALL2(__VA_ARGS__)

template<class T>
__global__ void template_kernel(T t){
    __shared__ T t2;
}

int foo31(){
  //CHECK: VA_CALL(([&] {
  //CHECK-NEXT:   dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
  //CHECK-NEXT:     sycl::local_accessor<int, 0> t2_acc_ct1(cgh);
  //CHECK:     cgh.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           template_kernel<int>(10, t2_acc_ct1);
  //CHECK-NEXT:         });
  //CHECK-NEXT:   });
  //CHECK-NEXT: }));
  VA_CALL( ([&]{ template_kernel<int><<<1,1,0>>>(10); }) );
}

class ArgClass{};

//CHECK: #define SIZE 256
#define SIZE 256
//CHECK: #define VACALL4(...) __VA_ARGS__()
//CHECK-NEXT: #define VACALL3(...) VACALL4(__VA_ARGS__)
//CHECK-NEXT: #define VACALL2(...) VACALL3(__VA_ARGS__)
//CHECK-NEXT: #define VACALL(x)                                                              \
//CHECK-NEXT:   dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {                   \
//CHECK-NEXT:     int i_ct0 = i;                                                            \
//CHECK-NEXT:     auto ac_ct0 = ac;                                                          \
//CHECK:     cgh.parallel_for(                                                          \
//CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2) *                            \
//CHECK-NEXT:                               sycl::range<3>(1, 1, SIZE),                      \
//CHECK-NEXT:                           sycl::range<3>(1, 1, SIZE)),                         \
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) { foo32(i_ct0, ac_ct0); });             \
//CHECK-NEXT:   });
#define VACALL4(...) __VA_ARGS__()
#define VACALL3(...) VACALL4(__VA_ARGS__)
#define VACALL2(...) VACALL3(__VA_ARGS__)
#define VACALL(x) foo32<<<2,SIZE,0>>>(i, ac)
__global__ void foo32(int a, ArgClass ac){}

// CHECK: int foo33(){
// CHECK-NEXT:   ArgClass ac;
// CHECK-NEXT:   int i;
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1038:{{[0-9]+}}: When the kernel function name is used as a macro argument, the
// CHECK-NEXT:   migration result may be incorrect. You need to verify the definition of the
// CHECK-NEXT:   macro.
// CHECK-NEXT:   */
// CHECK-NEXT:   VACALL2([&] {VACALL(0);
// CHECK-NEXT:   });
// CHECK-NEXT: }
int foo33(){
  ArgClass ac;
  int i;
  VACALL2([&]{VACALL(0);});
}


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>

void foo34() {

  int *ptr;
  thrust::host_vector<int> h_keys, h_values;
  thrust::device_vector<int> d_keys, d_values;
  thrust::equal_to<int> binary_pred;

  auto dummy_dev = thrust::device_ptr<int>(ptr);
  int numel = 1;
  using index_t = int;
  VACALL3([&]() {
    int64_t num_of_segments;
    {
      auto sorted_indices_dev = thrust::device_ptr<index_t>(ptr);
      auto dummy_dev = thrust::device_ptr<index_t>(ptr);
// CHECK:      auto ends =
// CHECK-NEXT: dpct::unique_copy(oneapi::dpl::execution::make_device_policy(
// CHECK-NEXT:                       dpct::get_in_order_queue()),
// CHECK-NEXT:                   sorted_indices_dev, sorted_indices_dev + numel,
// CHECK-NEXT:                   dpct::make_counting_iterator(0), dummy_dev,
// CHECK-NEXT:                   dpct::device_pointer<index_t>(ptr));
      auto ends = thrust::unique_by_key_copy(
          thrust::device, sorted_indices_dev, sorted_indices_dev + numel,
          thrust::make_counting_iterator(0), dummy_dev,
          thrust::device_ptr<index_t>(ptr));
    }
  });
}


//CHECK: #define ReturnErrorFunction                                                    \
//CHECK-NEXT:   int amax(dpct::blas::descriptor_ptr handle, const int n, const float *X,     \
//CHECK-NEXT:            const int incX, int &result) try {                                  \
//CHECK-NEXT:     return cublasIsamax(handle, n, (const float *)X, incX, &result);           \
//CHECK-NEXT:   }                                                                            \
//CHECK-NEXT:   catch (sycl::exception const &exc) {                                         \
//CHECK-NEXT:     std::cerr << exc.what() << "Exception caught at file:" << __FILE__         \
//CHECK-NEXT:               << ", line:" << __LINE__ << std::endl;                           \
//CHECK-NEXT:     std::exit(1);                                                              \
//CHECK-NEXT:   }

#define ReturnErrorFunction                                                         \
  cublasStatus_t amax( cublasHandle_t handle,                                       \
                       const int n, const float* X, const int incX, int& result )   \
  {                                                                                 \
    return cublasIsamax(handle, n, (const float*) X, incX, &result);                \
  }

ReturnErrorFunction

#define CUSOLVER_CHECK(err)                                                    \
  do {                                                                         \
    cusolverStatus_t err_ = (err);                                             \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                                     \
      printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);        \
      throw std::runtime_error("cusolver error");                              \
    }                                                                          \
  } while (0)

void foo35() {
  cusolverDnHandle_t handle;
  const int m = 3;
  double *d_A;
  const int lda = m;
  int lwork = 0;
  //CHECK: CUSOLVER_CHECK(DPCT_CHECK_ERROR(
  //CHECK-NEXT:   lwork = oneapi::mkl::lapack::geqrf_scratchpad_size<double>(*handle, m, m,
  //CHECK-NEXT:                                                              lda)));
  CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(handle, m, m, d_A, lda, &lwork));
}

#undef CUSOLVER_CHECK

template<class T>
class TemplateClass{};

__global__
template<class a>
__global__ void templatefoo3(){}

//CHECK: #define CALLTEMPLATEFOO                                                        \
//CHECK-NEXT:   q_ct1.parallel_for(                                                          \
//CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),     \
//CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {                                         \
//CHECK-NEXT:         templatefoo3<TemplateClass<TemplateClass<int>>>();                     \
//CHECK-NEXT:       });
//CHECK-NEXT: #define CALLTEMPLATEFOO2                                                       \
//CHECK-NEXT:   q_ct1.parallel_for(                                                          \
//CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),     \
//CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { templatefoo3<TemplateClass<int>>(); });

#define CALLTEMPLATEFOO templatefoo3<TemplateClass<TemplateClass<int>>><<<1,1,0>>>()
#define CALLTEMPLATEFOO2 templatefoo3<TemplateClass<int>><<<1,1,0>>>()
void foo35() {
  CALLTEMPLATEFOO;
  CALLTEMPLATEFOO2;
}