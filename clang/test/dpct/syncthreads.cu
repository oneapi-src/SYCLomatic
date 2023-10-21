// RUN: dpct --format-range=none -out-root %T/syncthreads %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/syncthreads/syncthreads.dp.cpp

#include "cuda_fp16.h"

// CHECK: void test_syncthreads(int *arr, const sycl::nd_item<3> &[[ITEMNAME:item_ct1]]) {
__global__ void test_syncthreads(int *arr) {
  // CHECK: [[ITEMNAME]].barrier(sycl::access::fence_space::local_space);
  __syncthreads();
  arr[threadIdx.x] = threadIdx.x;
}

// TODO: Support __synthreads_count, __syncthreads_and and __syncthreads_or
// __global__ void test_syncthreads_count(int *arr) {
//   int v = __syncthreads_count(threadIdx.x % 4);
//   arr[threadIdx.x] = v;
// }

// __global__ void test_syncthreads_and(int *arr) {
//   int v = __syncthreads_and(threadIdx.x % 4);
//   arr[threadIdx.x] = v;
// }

// __global__ void test_syncthreads_or() {
//   int v = __syncthreads_or(threadIdx.x % 4);
//   arr[threadIdx.x] = v;
// }

// TODO: Support __synthreads_count, __syncthreads_and and __syncthreads_or
//       Check local memory allocation while kernel invocation
// void test_kernel_call(int *arr) {
//   const size_t blocks_per_grid = 1;
//   const size_t threads_per_block = 32;
//
//   int *d_arr = NULL;
//   cudaMalloc((void **)&d_arr,
//              blocks_per_grid * threads_per_block * sizeof(int));
//
//   test_syncthreads<<<blocks_per_grid, threads_per_block>>>(d_arr);
//   test_syncthreads_count<<<blocks_per_grid, threads_per_block>>>(d_arr);
//   test_syncthreads_and<<<blocks_per_grid, threads_per_block>>>(d_arr);
//   test_syncthreads_or<<<blocks_per_grid, threads_per_block>>>(d_arr);
// }

// TODO: Further refine the analysis of barrier to support this case.
#define MACRO 1
__global__ void test1(unsigned int *ptr1, unsigned int *ptr2, unsigned int *ptr3) {
  uint4 mmm;
  for (;;) {
    for (;;) {
      ptr3[1];
    }
    for (;;) {
      for (;;) {
        mmm.x &= ptr1[MACRO];
        mmm.y &= ptr1[MACRO + 1];
        mmm.z &= ptr1[MACRO + 2];
        mmm.w &= ptr1[MACRO + 3];
      }
      int nnn = 0;
      nnn += __popc(mmm.x) + __popc(mmm.y) + __popc(mmm.z) + __popc(mmm.w);
    }
  }
  //     CHECK:  /*
  //CHECK-NEXT:  DPCT1065:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
  //CHECK-NEXT:  */
  //CHECK-NEXT:  item_ct1.barrier();
  //CHECK-NEXT:  for (;;) {
  //CHECK-NEXT:    /*
  //CHECK-NEXT:    DPCT1065:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
  //CHECK-NEXT:    */
  //CHECK-NEXT:    item_ct1.barrier();
  __syncthreads();
  for (;;) {
    __syncthreads();
  }
  if (true)
    for (;;) {
      unsigned int index = 1;
      atomicAdd(&ptr2[index], 1);
    }
}
#undef MACRO

__global__ void test2(float *ptr1, float *ptr2) {
  const int aaa(threadIdx.z);
  for (;;) {
    ptr1[aaa];
    // CHECK:item_ct1.barrier(sycl::access::fence_space::local_space);
    __syncthreads();
#pragma unroll
    for (;;) {}
    // CHECK:item_ct1.barrier(sycl::access::fence_space::local_space);
    __syncthreads();
  }
  ptr2[aaa];
}

// Unsupport label and goto stmt
__global__ void test3() {
  int a;
  int b;
  goto label;
  //CHECK:item_ct1.barrier();
  __syncthreads();
  a++;
label:
  b++;
}

// Unsupport MemberExpr whose type is pointer
struct S1 {
  float *data;
};
__global__ void test4(S1 s1) {
  s1.data;
  // CHECK:item_ct1.barrier();
  __syncthreads();
}

struct S2 {
  float data;
};
__global__ void test5(S2 s2) {
  s2.data;
  // CHECK:item_ct1.barrier(sycl::access::fence_space::local_space);
  __syncthreads();
}

// Unsupport MemberExpr whose type is array
struct S3 {
  float data[10];
};
__global__ void test7(S3 s3) {
  s3.data;
  // CHECK:item_ct1.barrier();
  __syncthreads();
}

__global__ void test8(S3 *s3) {
  int a = 1;
  s3[a].data;
  // CHECK:item_ct1.barrier();
  __syncthreads();
}

// Unsupport c++ constructor
__device__ void process_data(float*, float*) {}

struct S4 {
  float *data;
  __device__ S4(S4 &a) {
    process_data(data, a.data);
  }
};

__global__ void test9(S4 a) {
  S4 b(a);
  // CHECK:item_ct1.barrier();
  __syncthreads();
}

extern __shared__ float extern_local_decl[];

__global__ void test10(float *arg1_ptr, int arg2_scalar, int arg3_scalar) {
  int var1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (var1 < arg3_scalar) {
    float *var2 = extern_local_decl;
    var2[var1] = 0;
    if (var1 < 123) {
      float a = __expf(1.f);
      var2[var1] = a;
      arg1_ptr[var1] = a;
    }
    // CHECK: /*
    // CHECK-NEXT: DPCT1118:{{[0-9]+}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1113:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with sycl::nd_item::barrier() if function "test10" is called in a multidimensional kernel.
    // CHECK-NEXT: */
    // CHECK-NEXT: item_ct1.barrier(sycl::access::fence_space::local_space);
    // CHECK-NEXT: int var3 = arg3_scalar / 2;
    __syncthreads();
    int var3 = arg3_scalar / 2;
    while (var3 != 0) {
      if (var1 < var3) {
        var2[var1] += var2[var1 + var3];
      }
      // CHECK: var3 = var3 / 2;
      // CHECK-NEXT: /*
      // CHECK-NEXT: DPCT1118:{{[0-9]+}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
      // CHECK-NEXT: */
      // CHECK-NEXT: /*
      // CHECK-NEXT: DPCT1113:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with sycl::nd_item::barrier() if function "test10" is called in a multidimensional kernel.
      // CHECK-NEXT: */
      // CHECK-NEXT: item_ct1.barrier(sycl::access::fence_space::local_space);
      var3 = var3 / 2;
      __syncthreads();
    }
    float var4 = var2[0] + 1.f;
    if (var1 < arg2_scalar) {
      arg1_ptr[var1] /= var4;
      arg1_ptr[var1] = sqrtf(arg1_ptr[var1]);
    }
  }
  // CHECK:   /*
  // CHECK-NEXT:   DPCT1113:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with sycl::nd_item::barrier() if function "test10" is called in a multidimensional kernel.
  // CHECK-NEXT:   */
  // CHECK-NEXT:   item_ct1.barrier(sycl::access::fence_space::local_space);
  // CHECK-NEXT: }
  __syncthreads();
}

int query_block(const int x) {
  return x;
}

void foo() {
  float *f;
  int a;
  int n = 128;
  dim3 block(n);
  dim3 grid(query_block(n));
  test10<<<grid, block>>>(f, a, a);
}

__global__ void test11(float *a_ptr, float *b_ptr,
                       float *c_ptr, float *d_ptr,
                       int const e_scalar, float *f_ptr,
                       float *g_ptr, float const h_scalar,
                       int *i_ptr, size_t const j_scalar) {
  int const eight = 8;
  const float *a_c_ptr = a_ptr;
  const float *b_c_ptr = b_ptr;
  float *nc_c_ptr = c_ptr;
  const float *d_c_ptr = d_ptr;
  const float *f_c_ptr = f_ptr;
  const float *g_c_ptr = g_ptr;
  const float h_c_scalar = h_scalar;
  const int *i_c_ptr = i_ptr;
  const int e_c_scalar = e_scalar;
  const int zero = 0;

  __shared__ float local[123];
  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

  int idx = idx_x + eight + (idx_y * e_c_scalar + eight) * j_scalar;
  size_t local_idx_x = threadIdx.x;

  size_t var0 = (local_idx_x + eight);
  float var1[321];
  float var2[123];
  int var3 = i_c_ptr[321 - 123];
  float var4[321 + 123];
  float var5[321];
  for (unsigned int iter = 0; iter <= 321; iter++) {
    var4[iter] = a_c_ptr[idx + j_scalar * iter];
  }
  for (unsigned int iter = 1; iter <= 321; iter++) {
    var5[iter - 1] = a_c_ptr[idx - j_scalar * iter];
    var1[iter - 1] = f_c_ptr[iter - 1];
    var2[iter - 1] = g_c_ptr[iter - 1];
  }
  bool var6 = false;
  if (local_idx_x < 123)
    var6 = true;
  const unsigned int var7 = blockDim.x;
  for (int i = 0; i < e_c_scalar; i++) {
    local[var0] = var4[0];
    if (var6) {
      local[var0 - 123] = a_c_ptr[idx - 123];
      local[var0 + var7] = a_c_ptr[idx + var7];
    }
    // CHECK: DPCT1118:{{[0-9]+}}: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1113:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier(sycl::access::fence_space::local_space) with sycl::nd_item::barrier() if function "test11" is called in a multidimensional kernel.
    // CHECK-NEXT: */
    // CHECK-NEXT: (item_ct1.get_local_range(2) < j_scalar) ? item_ct1.barrier(sycl::access::fence_space::local_space) : item_ct1.barrier();
    // CHECK-NEXT: float var8 = 0;
    __syncthreads();
    float var8 = 0;
    var8 = fmaf(local[var0], h_c_scalar, var8);
    for (int iter = 1; iter <= 8; iter++) {
      var8 = fmaf(local[var0 - iter], var1[iter - 1], var8);
      var8 = fmaf(local[var0 + iter], var1[iter - 1], var8);
    }
    for (int iter = 1; iter <= 123; iter++) {
      var8 = fmaf(var4[iter], var2[iter - 1], var8);
      var8 = fmaf(var5[iter - 1], var2[iter - 1], var8);
    }
    var8 = fmaf(d_c_ptr[idx], var8, -b_c_ptr[idx]);
    var8 = fmaf(2.0f, local[var0], var8);
    nc_c_ptr[idx] = var8;
    idx += j_scalar;
    for (unsigned int iter = 321 - 123; iter > 0; iter--) {
      var5[iter] = var5[iter - 1];
    }
    var5[0] = var4[0];
    for (unsigned int iter = 0; iter < 321; iter++) {
      var4[iter] = var4[iter + 1];
    }
    var4[8] = a_c_ptr[idx + var3];
  }
}

typedef unsigned char uint8_t;

__device__ int bar12(int num) {
  int n = num - 1;
  n |= n >> 1;
  return n + 1;
}

__global__ void test12(uint8_t *pout) {
  __shared__ float local[123];
  int idx = 456;
  idx = bar12(idx);
  local[idx] = idx;
  //CHECK:  item_ct1.barrier(sycl::access::fence_space::local_space);
  //CHECK-NEXT:  pout[idx] = local[idx];
  __syncthreads();
  pout[idx] = local[idx];
}

__device__ int d_a[10];

__device__ int bar13(int num) {
  int n = num - 1;
  n |= d_a[1] >> 1;
  return n + 1;
}

// TODO: Further refine the analysis of barrier to support this case.
__global__ void test13(uint8_t *pout) {
  int idx = 456;
  idx = bar13(idx);
  //CHECK:  pout[idx] = 456;
  //CHECK-NEXT:  /*
  //CHECK-NEXT:  DPCT1065:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
  //CHECK-NEXT:  */
  //CHECK-NEXT:  item_ct1.barrier();
  //CHECK-NEXT:  uint8_t a = 4;
  pout[idx] = 456;
  __syncthreads();
  uint8_t a = 4;
  uint8_t b = abs(a);
}

__device__ void barbar14() {}

__device__ int bar14() {
  barbar14();
  int n = 123;
  return n + 1;
}

__global__ void test14(uint8_t *pout) {
  int idx = 456;
  idx = bar14();
  //CHECK:  pout[idx] = 789;
  //CHECK-NEXT:  item_ct1.barrier(sycl::access::fence_space::local_space);
  //CHECK-NEXT:  uint8_t a = 4;
  pout[idx] = 789;
  __syncthreads();
  uint8_t a = 4;
  uint8_t b = abs(a);
}

__global__ void test15(float *_res) {
  __shared__ float2 ker_even[123]; // shared mem
  //CHECK:  float *res = _res + 123;
  //CHECK-NEXT:  res[2] = 123;
  //CHECK-NEXT:  item_ct1.barrier(sycl::access::fence_space::local_space);
  //CHECK-NEXT:  res += 2;
  float *res = _res + 123;
  res[2] = 123;
  __syncthreads();
  res += 2;
}

template <class T> __global__ void test16(T *res) {
  //CHECK:  auto a = res[2];
  //CHECK-NEXT:  item_ct1.barrier(sycl::access::fence_space::local_space);
  //CHECK-NEXT:  a++;
  auto a = res[2];
  __syncthreads();
  a++;
}

template <class T> void foo16(int grid, int block, T *res, cudaStream_t stream) {
  test16<T><<<grid, block, 0, stream>>>(res);
}

template void foo16<half>(int grid, int block, half *res, cudaStream_t stream);
template void foo16<float>(int grid, int block, float *res, cudaStream_t stream);

__global__ void test17(float *f) {
  __shared__ float ff[10];
  //CHECK:  *((sycl::float2 *)&ff[2]) = *((sycl::float2 *)&f[123]);
  //CHECK-NEXT:  item_ct1.barrier(sycl::access::fence_space::local_space);
  item_ct1.barrier(sycl::access::fence_space::local_space);
  *((float2 *)&ff[2]) = *((float2 *)&f[123]);
  __syncthreads();
  float xyz = f[123];
  // other code ...
}

__global__ void test18(unsigned int *aaa) {
  __shared__ unsigned int bbb[2][10];
  //CHECK:  *((sycl::uint4 *)(&aaa[5])) = *((sycl::uint4 *)&bbb[1][2]);
  //CHECK-NEXT:  item_ct1.barrier(sycl::access::fence_space::local_space);
  *((uint4 *)(&aaa[5])) = *((uint4 *)&bbb[1][2]);
  __syncthreads();
  unsigned int xyz = bbb[1][2];
  // other code ...
}

__global__ void test19(unsigned int *a) {
  //CHECK: a[item_ct1.get_local_id(2)]++;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1065:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
  //CHECK-NEXT: */
  //CHECK-NEXT: item_ct1.barrier();
  a[threadIdx.x]++;
  __syncthreads();
  a[threadIdx.x]++;
}

__device__ void recursive1(int i);

__device__ void recursive(int i) {
  if (i > 0) {
    recursive1(i - 1);
  }
}

__device__ void recursive1(int i) {
  if (i > 0) {
    recursive(i - 1);
  }
}

__global__ void test20() {
  // CHECK: recursive(10);
  // CHECK-NEXT: item_ct1.barrier(sycl::access::fence_space::local_space);
  recursive(10);
  __syncthreads();
}
