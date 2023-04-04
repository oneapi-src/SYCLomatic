// RUN: dpct --format-range=none -out-root %T/syncthreads %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/syncthreads/syncthreads.dp.cpp

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
//     CHECK:  item_ct1.barrier(sycl::access::fence_space::local_space);
//CHECK-NEXT:  for (;;) {
//CHECK-NEXT:    item_ct1.barrier(sycl::access::fence_space::local_space);
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
__global__ void test3(float *f) {
  int a;
  int b;
  f[a] = f[b];
  goto label;
  //CHECK:item_ct1.barrier();
  __syncthreads();
  a++;
label:
  b++;
}

__global__ void test3_1() {
  int a;
  int b;
  goto label;
  //CHECK:item_ct1.barrier(sycl::access::fence_space::local_space);
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

extern __shared__ float cache[];

__global__ void test10(float *pdata, int k, int num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    float *pcache = cache;
    pcache[tid] = 0;
    if (tid < 123) {
      float a = __expf(1.f);
      pcache[tid] = a;
      pdata[tid] = a;
    }
    // CHECK: item_ct1.barrier(sycl::access::fence_space::local_space);
    // CHECK-NEXT: int half = num >> 1;
    __syncthreads();
    int half = num >> 1;
    while (half != 0) {
      if (tid < half) {
        pcache[tid] += pcache[tid + half];
      }
      // CHECK: half = half >> 1;
      // CHECK-NEXT: item_ct1.barrier(sycl::access::fence_space::local_space);
      half = half >> 1;
      __syncthreads();
    }
    float b = pcache[0] + 1.f;
    if (tid < k) {
      pdata[tid] /= b;
      pdata[tid] = sqrtf(pdata[tid]);
    }
  }
// CHECK:   item_ct1.barrier(sycl::access::fence_space::local_space);
// CHECK-NEXT: }
  __syncthreads();
}



__global__ void test11(float *a, float *b,
                              float *c, float *d,
                              int const e, float *f,
                              float *g, float const h,
                              int *i, size_t const j) {
  int const eight = 8;
  const float *a_c = a;
  const float *b_c = b;
  float *nc_c = c;
  const float *d_c = d;
  const float *f_c = f;
  const float *g_c = g;
  const float h_c = h;
  const int *i_c = i;
  const int e_c = e;
  const int zero = 0;

  __shared__ float local[123];
  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

  int idx = idx_x + eight + (idx_y * e_c + eight) * j;
  size_t local_idx_x = threadIdx.x;

  size_t var0 = (local_idx_x + eight);
  float var1[321];
  float var2[123];
  int var3 = i_c[321 - 123];
  float var4[321 + 123];
  float var5[321];
  for (unsigned int iter = 0; iter <= 321; iter++) {
    var4[iter] = a_c[idx + j * iter];
  }
  for (unsigned int iter = 1; iter <= 321; iter++) {
    var5[iter - 1] = a_c[idx - j * iter];
    var1[iter - 1] = f_c[iter - 1];
    var2[iter - 1] = g_c[iter - 1];
  }
  bool var6 = false;
  if (local_idx_x < 123)
    var6 = true;
  const unsigned int var7 = blockDim.x;
  for (int i = 0; i < e_c; i++) {
    local[var0] = var4[0];
    if (var6) {
      local[var0 - 123] = a_c[idx - 123];
      local[var0 + var7] = a_c[idx + var7];
    }
    // CHECK: item_ct1.barrier(sycl::access::fence_space::local_space);
    // CHECK-NEXT: float var8 = 0;
    __syncthreads();
    float var8 = 0;
    var8 = fmaf(local[var0], h_c, var8);
    for (int iter = 1; iter <= 8; iter++) {
      var8 = fmaf(local[var0 - iter], var1[iter - 1], var8);
      var8 = fmaf(local[var0 + iter], var1[iter - 1], var8);
    }
    for (int iter = 1; iter <= 123; iter++) {
      var8 = fmaf(var4[iter], var2[iter - 1], var8);
      var8 = fmaf(var5[iter - 1], var2[iter - 1], var8);
    }
    var8 = fmaf(d_c[idx], var8, -b_c[idx]);
    var8 = fmaf(2.0f, local[var0], var8);
    nc_c[idx] = var8;
    idx += j;
    for (unsigned int iter = 321 - 123; iter > 0; iter--) {
      var5[iter] = var5[iter - 1];
    }
    var5[0] = var4[0];
    for (unsigned int iter = 0; iter < 321; iter++) {
      var4[iter] = var4[iter + 1];
    }
    var4[8] = a_c[idx + var3];
  }
}

