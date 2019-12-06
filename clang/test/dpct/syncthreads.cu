// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/syncthreads.dp.cpp

// CHECK: void test_syncthreads(int *arr, cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void test_syncthreads(int *arr) {
  // CHECK: [[ITEMNAME]].barrier();
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
