// RUN: dpct --format-range=none -out-root %T/syncthreads %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/syncthreads/syncthreads.dp.cpp

// CHECK: void test_syncthreads(int *arr, sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
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
__device__ void process_data(float*, float*);

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