// RUN: cat %s > %T/macro_test.cu
// RUN: cd %T
// RUN: dpct -out-root %T macro_test.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/macro_test.dp.cpp --match-full-lines macro_test.cu

#define CUDA_NUM_THREADS 1024+32
#define GET_BLOCKS(n,t)  1+n+t-1
#define GET_BLOCKS2(n,t) 1+n+t
#define GET_BLOCKS3(n,t) n+t-1
#define GET_BLOCKS4(n,t) n+t

class DDD{
public:
  dim3* A;
  dim3 B;
};

#define EMPTY_MACRO(x) x
//CHECK:#define GET_MEMBER_MACRO(x) x[1] = 5
#define GET_MEMBER_MACRO(x) x.y = 5

__global__ void foo_kernel() {}

__global__ void foo2(){
  // CHECK: #define IMUL(a, b) sycl::mul24(a, b)
  // CHECK-NEXT: int vectorBase = IMUL(1, 2);
  #define IMUL(a, b) __mul24(a, b)
  int vectorBase = IMUL(1, 2);
}

__global__ void foo3(int x, int y) {}

void foo() {
  DDD d3;

  // CHECK: (*d3.A)[0] = 3;
  // CHECK-NEXT: d3.B[0] = 2;
  // CHECK-NEXT: EMPTY_MACRO(d3.B[0]);
  // CHECK-NEXT: GET_MEMBER_MACRO(d3.B);
  d3.A->x = 3;
  d3.B.x = 2;
  EMPTY_MACRO(d3.B.x);
  GET_MEMBER_MACRO(d3.B);

  int outputThreadCount = 512;

  // CHECK: dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS(outputThreadCount, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS(outputThreadCount, outputThreadCount), 2, 0>>>();

  // CHECK: dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS2(CUDA_NUM_THREADS, CUDA_NUM_THREADS), 0, 0>>>();

  // CHECK: dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 0),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 0)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
  // CHECK-NEXT: });
  foo_kernel<<<GET_BLOCKS3(CUDA_NUM_THREADS, outputThreadCount), 0, 0>>>();

  // CHECK: dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(
  // CHECK-NEXT:           sycl::range<3>(1, 1,
  // CHECK-NEXT:                          GET_BLOCKS4(outputThreadCount, CUDA_NUM_THREADS)) *
  // CHECK-NEXT:               sycl::range<3>(1, 1, 2),
  // CHECK-NEXT:           sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });
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

// CHECK: #define cbrt(x) pow((double)x,(double)(1.0/3.0))
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

//CHECK: #define MACRO_KC                                                               \
//CHECK-NEXT:   dpct::get_default_queue().submit([&](sycl::handler &cgh) {                   \
//CHECK-NEXT:     cgh.parallel_for(                                                          \
//CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2),   \
//CHECK-NEXT:                           sycl::range<3>(1, 1, 2)),                            \
//CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) { foo_kernel(); });                     \
//CHECK-NEXT:   });
#define MACRO_KC foo_kernel<<<2, 2, 0>>>();

//CHECK: MACRO_KC
MACRO_KC

// CHECK: #define HARD_KC(NAME)                                                          \
// CHECK-NEXT:   NAME<<<sycl::range<3>(2, 1, 1), sycl::range<3>(2, 1, 1), 0>>>();
#define HARD_KC(NAME) NAME<<<2,2,0>>>();
// CHECK: /*
// CHECK-NEXT: DPCT1038:0: Kernel calls with kernel function name in macro arguments are not
// CHECK-NEXT: supported. Try to rewrite this code using DPC++ kernel.
// CHECK-NEXT: */
//CHECK-NEXT: HARD_KC(foo_kernel)
HARD_KC(foo_kernel)


// CHECK: #define MACRO_KC2(a, b, c, d)                                                  \
// CHECK-NEXT:   dpct::get_default_queue().submit([&](sycl::handler &cgh) {                   \
// CHECK-NEXT:     auto dpct_global_range = a * b;                                            \
// CHECK-NEXT:                                                                                \
// CHECK-NEXT:     auto c_ct0 = c;                                                            \
// CHECK-NEXT:     auto d_ct1 = d;                                                            \
// CHECK-NEXT:                                                                                \
// CHECK-NEXT:     cgh.parallel_for(                                                          \
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),             \
// CHECK-NEXT:                                          dpct_global_range.get(1),             \
// CHECK-NEXT:                                          dpct_global_range.get(0)),            \
// CHECK-NEXT:                           sycl::range<3>(b.get(2), b.get(1), b.get(0))),       \
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) { foo3(c_ct0, d_ct1); });               \
// CHECK-NEXT:   });
#define MACRO_KC2(a,b,c,d) foo3<<<a, b, 0>>>(c,d);

dim3 griddim = 2;
dim3 threaddim = 32;

// CHECK: MACRO_KC2(griddim,threaddim,1,0)
MACRO_KC2(griddim,threaddim,1,0)

// [Note] Since 3 and 2 are migrated to sycl::range<3>, if they are used in macro as native numbers,
// there might be some issues in the migrated code.
// Since this is a corner case, not to emit warning message here.
// CHECK: MACRO_KC2(sycl::range<3>(3, 1, 1), sycl::range<3>(2, 1, 1), 1, 0)
MACRO_KC2(3,2,1,0)

// CHECK: MACRO_KC2(sycl::range<3>(5, 4, 3), sycl::range<3>(2, 1, 1), 1, 0)
MACRO_KC2(dim3(5,4,3),2,1,0)

}


