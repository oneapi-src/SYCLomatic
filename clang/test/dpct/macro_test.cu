// RUN: cat %s > %T/macro_test.cu
// RUN: cat %S/macro_test.h > %T/macro_test.h
// RUN: cd %T
// RUN: rm -rf %T/macro_test_output
// RUN: mkdir %T/macro_test_output
// RUN: dpct -out-root %T/macro_test_output macro_test.cu --cuda-include-path="%cuda-path/include" --stop-on-parse-err -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/macro_test_output/macro_test.dp.cpp --match-full-lines macro_test.cu
// RUN: FileCheck --input-file %T/macro_test_output/macro_test.h --match-full-lines macro_test.h
#include <math.h>
#include "macro_test.h"

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
//CHECK-NEXT:   , sycl::nd_item<3> item_ct1) {
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
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
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
  //CHECK-NEXT: CALL((q_ct1.submit([&](sycl::handler &cgh) {
  //CHECK:   cgh.parallel_for(sycl::nd_range<3>(x * x, x),
  //CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:         foo_kernel();
  //CHECK-NEXT:       });
  //CHECK-NEXT: });))
  CALL( (foo_kernel<<<1, 2, 0>>>()) )

  //CHECK: #define AA 3
  //CHECK-NEXT: #define MCALL                                                                  \
  //CHECK-NEXT: q_ct1.submit([&](sycl::handler &cgh) {                                       \
  //CHECK-NEXT:   cgh.parallel_for(                                                          \
  //CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * 2 * AA, 2 * AA),           \
  //CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });                     \
  //CHECK-NEXT: });
  //CHECK-NEXT: MCALL
  #define AA 3
  #define MCALL foo_kernel<<<dim3(2,1), 2*AA, 0>>>();
  MCALL

  // CHECK: q_ct1.submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS(outputThreadCount, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel();
  // CHECK-NEXT:       });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS(outputThreadCount, outputThreadCount), 2, 0>>>();

  // CHECK: q_ct1.submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel();
  // CHECK-NEXT:       });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS), 0, 0>>>();

  // CHECK: q_ct1.submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel();
  // CHECK-NEXT:       });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount), 0, 0>>>();

  // CHECK: q_ct1.submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS4(outputThreadCount, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         foo_kernel();
  // CHECK-NEXT:       });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS4(outputThreadCount, CUDA_NUM_THREADS), 2, 0>>>();

  // Test if SIGABRT.
  // No check here because the generated code need further fine tune.
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

// CHECK: #define NNBI(x) floor(x + 0.5)
// CHECK-NEXT: NNBI(3.0);
#define NNBI(x) floor(x+0.5)
NNBI(3.0);

// CHECK: #define PI acos(-1)
#define PI acos(-1)
// CHECK: double cosine = cos(2 * PI);
double cosine = cos(2 * PI);

//CHECK: #define MACRO_KC                                                                    \
//CHECK-NEXT:   q_ct1.submit([&](sycl::handler &cgh) {                                       \
//CHECK-NEXT:     cgh.parallel_for(                                                          \
//CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2),   \
//CHECK-NEXT:                           sycl::range<3>(1, 1, 2)),                            \
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });                     \
//CHECK-NEXT:   });
#define MACRO_KC foo_kernel<<<2, 2, 0>>>();

//CHECK: MACRO_KC
MACRO_KC


//CHECK: #define HARD_KC(NAME, a, b, c, d)                                                   \
//CHECK-NEXT:   q_ct1.submit([&](sycl::handler &cgh) {                                       \
//CHECK-NEXT:     auto c_ct0 = c;                                                            \
//CHECK-NEXT:     auto d_ct1 = d;                                                            \
//CHECK-NEXT:                                                                                \
//CHECK-NEXT:     cgh.parallel_for(sycl::nd_range<3>(a * b, b),                              \
//CHECK-NEXT:                      [=](sycl::nd_item<3> item_ct1) { foo3(c_ct0, d_ct1); });  \
//CHECK-NEXT:   });
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1038:{{[0-9]+}}: When the kernel function name is used as a macro argument, the
//CHECK-NEXT:   migration result may be incorrect. You need to verify the definition of the
//CHECK-NEXT:   macro.
//CHECK-NEXT:   */
//CHECK-NEXT:   HARD_KC(foo3, sycl::range<3>(1, 1, 3), sycl::range<3>(1, 1, 2), 1, 0)
#define HARD_KC(NAME,a,b,c,d) NAME<<<a,b,0>>>(c,d);
HARD_KC(foo3,3,2,1,0)


//CHECK: #define MACRO_KC2(a, b, c, d)                                                       \
//CHECK-NEXT:   q_ct1.submit([&](sycl::handler &cgh) {                                       \
//CHECK-NEXT:     auto c_ct0 = c;                                                            \
//CHECK-NEXT:     auto d_ct1 = d;                                                            \
//CHECK-NEXT:                                                                                \
//CHECK-NEXT:     cgh.parallel_for(sycl::nd_range<3>(a * b, b),                  \
//CHECK-NEXT:                      [=](sycl::nd_item<3> item_ct1) { foo3(c_ct0, d_ct1); });  \
//CHECK-NEXT:   });
#define MACRO_KC2(a,b,c,d) foo3<<<a, b, 0>>>(c,d);

dim3 griddim = 2;
dim3 threaddim = 32;

// CHECK: MACRO_KC2(griddim,threaddim,1,0)
MACRO_KC2(griddim,threaddim,1,0)

// [Note] Since 3 and 2 are migrated to sycl::range<3>, if they are used in macro as native numbers,
// there might be some issues in the migrated code.
// Since this is a corner case, not to emit warning message here.
// CHECK: MACRO_KC2(sycl::range<3>(1, 1, 3), sycl::range<3>(1, 1, 2), 1, 0)
MACRO_KC2(3,2,1,0)

// CHECK: MACRO_KC2(sycl::range<3>(3, 4, 5), sycl::range<3>(1, 1, 2), 1, 0)
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

  //CHECK: q_ct1.submit([&](sycl::handler &cgh) {
  //CHECK-NEXT:   cgh.parallel_for(
  //CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2),
  //CHECK-NEXT:                         sycl::range<3>(1, 1, 2)),
  //CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:         foo_kernel2(3, 3, item_ct1);
  //CHECK-NEXT:       });
  //CHECK-NEXT: });
  foo_kernel2<<<2, 2, 0>>>(3,3
    #ifdef MACRO_CC
    , 2
    #endif
  );

  #define SIZE    (100*1024*1024)
  unsigned char *dev_buffer;
  unsigned char *buffer = (unsigned char*)malloc(500);
  //CHECK: q_ct1.memcpy(dev_buffer, buffer, SIZE).wait();
  cudaMemcpy( dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);
}

#define MMM(x)
texture<float4, 1, cudaReadModeElementType> table;
__global__ void foo4(){
  float r2 = 2.0;
  MMM( float rsqrtfr2; );
  // CHECK: sycl::float4 f4 = table.read(MMM(rsqrtfr2 =) sycl::rsqrt(r2) MMM(== 0));
  float4 f4 = tex1D(table, MMM(rsqrtfr2 =) rsqrtf(r2) MMM(==0));
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

// CHECK: #define CCC AAA, float *sp_lj, float *sp_coul, int *ljd, dpct::accessor<double, dpct::local, 2> la, int *b=0
// CHECK-NEXT: #define CC AAA, BBB
#define CCC AAA, int *b=0
#define CC AAA, BBB

// CHECK: #define CCCC(x) void fooc(x)
// CHECK-NEXT: #define CCCCC(x) void foocc(x, float *sp_lj, float *sp_coul, int *ljd, dpct::accessor<double, dpct::local, 2> la)
#define CCCC(x) __device__ void fooc(x)
#define CCCCC(x) __device__ void foocc(x)

// CHECK: #define XX(x) void foox(x, float *sp_lj, float *sp_coul, int *ljd, dpct::accessor<double, dpct::local, 2> la)
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
  __shared__ int ljd[0];
  __shared__ double la[8][0];
}

// CHECK: CCCCC(int *a)
// CHECK-NEXT: {
// CHECK-NEXT: }
CCCCC(int *a)
{
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[0];
  __shared__ double la[8][0];
}


// CHECK: CCCC(CCC)
// CHECK-NEXT: {
// CHECK-NEXT: }
CCCC(CCC)
{
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[0];
  __shared__ double la[8][0];
}

// CHECK: #define FFF void foo(AAA, BBB, float *sp_lj, float *sp_coul, int *ljd, dpct::accessor<double, dpct::local, 2> la)
#define FFF __device__ void foo(AAA, BBB)

// CHECK: FFF
// CHECK-NEXT: {
// CHECK-NEXT: }
FFF
{
  __shared__ float sp_lj[4];
  __shared__ float sp_coul[4];
  __shared__ int ljd[0];
  __shared__ double la[8][0];

}

// CHECK: #define FFFFF(aaa,bbb) void foo4(const int * __restrict__ aaa, const float * __restrict__ bbb, int *c, BBB, sycl::nd_item<3> item_ct1, float *sp_lj, float *sp_coul, int *ljd, dpct::accessor<double, dpct::local, 2> la)
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
  __shared__ int ljd[0];
  __shared__ double la[8][0];
  const int tid = threadIdx.x;
}

// CHECK: #define FFFFFF(aaa,bbb) void foo5(const int * __restrict__ aaa, const float * __restrict__ bbb, sycl::nd_item<3> item_ct1, float *sp_lj, float *sp_coul, int *ljd, dpct::accessor<double, dpct::local, 2> la)
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
  __shared__ int ljd[0];
  __shared__ double la[8][0];
  const int tid = threadIdx.x;
}

// CHECK: void foo6(AAA, BBB, float *sp_lj, float *sp_coul, int *ljd,
// CHECK-NEXT:   dpct::accessor<double, dpct::local, 2> la)
// CHECK-NEXT: {
// CHECK-NEXT: }
__device__ void foo6(AAA, BBB)
{
   __shared__ float sp_lj[4];
   __shared__ float sp_coul[4];
   __shared__ int ljd[0];
   __shared__ double la[8][0];
}


//CHECK: #define MM __umul24
//CHECK-NEXT: #define MUL(a, b) sycl::mul24((unsigned int)a, (unsigned int)b)
//CHECK-NEXT: void foo7(sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:   unsigned int tid =
//CHECK-NEXT:       MUL(item_ct1.get_local_range().get(2), item_ct1.get_group(2)) +
//CHECK-NEXT:       item_ct1.get_local_range().get(2);
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
//CHECK-NEXT:   SLOW(dpct::atomic_fetch_add(&data[0], (double)tid);
//CHECK-NEXT:         dpct::atomic_fetch_add(&data[1], (double)(tid + 1));
//CHECK-NEXT:         dpct::atomic_fetch_add(&data[2], (double)(tid + 2)););
//CHECK-NEXT: }
__global__ void foo8(){
#define SLOW(X) X
  double* data;
  unsigned long long int tid;
  SLOW(atomicAdd(&data[0], tid);
  atomicAdd(&data[1], tid + 1);
  atomicAdd(&data[2], tid + 2););
}


//CHECK: #define DFABS(x) (double)sycl::fabs((x))
//CHECK-NEXT: #define MAX(x, y) sycl::max(x, y)
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
//CHECK-NEXT:   dpct::get_default_queue().submit([&](sycl::handler &cgh) {                   \
//CHECK-NEXT:     cgh.parallel_for(                                                          \
//CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),   \
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) { templatefoo<A, B>(); });              \
//CHECK-NEXT:   });
//CHECK-NEXT: #define CALL_KERNEL(C, D) KERNEL(C, D); int a = 0;
//CHECK-NEXT: #define CALL_KERNEL2(E, F) sycl::range<3>(1, 1, 1)
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

//CHECK: void foo11(sycl::nd_item<3> item_ct1){
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
//CHECK-NEXT: #define POW(x, y) sycl::pow(x, y)
//CHECK-NEXT: #define SQRT(x) sycl::sqrt(x)
//CHECK-NEXT: void foo12(){
//CHECK-NEXT: real *vx;
//CHECK-NEXT: real *vy;
//CHECK-NEXT: int id;
//CHECK-NEXT: real v2 = SQRT(SQRT(POW(vx[id], 2.0) + POW(vy[id], 2.0)));
//CHECK-NEXT: }
typedef float real;
#define POW(x,y)    powf(x,y)
#define SQRT(x)     sqrtf(x)
__global__ void foo12(){
real *vx;
real *vy;
int id;
real v2 = SQRT(SQRT(POW(vx[id], 2.0) + POW(vy[id], 2.0)));
}