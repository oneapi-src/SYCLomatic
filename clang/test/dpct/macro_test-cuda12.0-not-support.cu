// UNSUPPORTED: cuda-8.0, cuda-12.0, cuda-12.1
// UNSUPPORTED: v8.0, v12.0, v12.1
// RUN: cat %s > %T/macro_test-cuda12.0-not-support.cu
// RUN: cd %T
// RUN: rm -rf %T/macro_test-cuda12.0-not-support_output
// RUN: mkdir %T/macro_test-cuda12.0-not-support_output
// RUN: dpct -out-root %T/macro_test-cuda12.0-not-support_output macro_test-cuda12.0-not-support.cu --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/macro_test-cuda12.0-not-support_output/macro_test-cuda12.0-not-support.dp.cpp --match-full-lines macro_test-cuda12.0-not-support.cu
#include <math.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <algorithm>

#include <stdio.h>

// CHECK: #include <algorithm>

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
#define CALL(x) x;

#define MMM(x)
texture<float4, 1, cudaReadModeElementType> table;
__global__ void foo4(){
  float r2 = 2.0;
  MMM( float rsqrtfr2; );
  // CHECK: sycl::float4 f4 = table.read(MMM(rsqrtfr2 =) sycl::rsqrt(r2) MMM(== 0));
  float4 f4 = tex1D(table, MMM(rsqrtfr2 =) rsqrtf(r2) MMM(==0));
}

//CHECK: void foo15(){
//CHECK-NEXT:   /*
//CHECK-NEXT:   DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
//CHECK-NEXT:   */
//CHECK-NEXT:   dpct::image_wrapper<float, 1> aaa;
//CHECK-NEXT:   float *f_a = NULL;
//CHECK-NEXT:   CALL(aaa.attach(f_a, CUDA_NUM_THREADS * sizeof(int)))
//CHECK-NEXT: }
void foo15(){
  texture<float, 1, cudaReadModeElementType> aaa;
  float *f_a = NULL;
  CALL(cudaBindTexture(0, aaa, f_a, CUDA_NUM_THREADS * sizeof(int)))
}

//     CHECK: #define CBTTA(aa, bb) do {                                                     \
//CHECK-NEXT:     CALL(aa.attach(bb));                                                       \
//CHECK-NEXT:   } while (0)
#define CBTTA(aa,bb) do {                 \
  CALL(cudaBindTextureToArray(aa, bb));   \
} while(0)

//     CHECK: #define CBTTA2(aa, bb, cc) do {                                                \
//CHECK-NEXT:     CALL(aa.attach(bb, cc));                                                   \
//CHECK-NEXT:   } while (0)
#define CBTTA2(aa,bb,cc) do {                 \
  CALL(cudaBindTextureToArray(aa, bb, cc));   \
} while(0)

//CHECK: void foo19(){
//CHECK-NEXT:   dpct::image_wrapper<sycl::float4, 2> tex42;
//CHECK-NEXT:   dpct::image_matrix_p a42;
//CHECK-NEXT:   CBTTA(tex42,a42);
//CHECK-NEXT:   CBTTA2(tex42, a42, tex42.get_channel());
//CHECK-NEXT: }
void foo19(){
  texture<float4, 2> tex42;
  cudaArray_t a42;
  CBTTA(tex42,a42);
  CBTTA2(tex42,a42,tex42.channelDesc);
}
