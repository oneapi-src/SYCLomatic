// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/blocklevel/blockload %S/blockload.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockscan/blockload.dp.cpp --match-full-lines %s

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32

const int N = 256;
const int BlockSize = 128;
const int ItemsPerThread = 4;


void init_data(int* data, int num) {
  for(int i = 0; i < num; i++)
    data[i] = i;
}
void verify_data(int* data, int num) {
  return;
}
void print_data(int* data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

//CHECK: void SumKernel(int* data,
//CHECK-NEXT:  const sycl::nd_item<3> &item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT:  int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT:  int input = data[threadid];
//CHECK-NEXT:  int output = 0;
//CHECK-NEXT:  output = sycl::reduce_over_group(item_ct1.get_group(), input, sycl::plus<>());
//CHECK-NEXT:  data[threadid] = output;
//CHECK-NEXT:}

__global__ void BlockLoadKernel(int *d_data)
{
    // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
    typedef cub::BlockLoad<int, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;
    // Allocate shared memory for BlockLoad
    __shared__ typename BlockLoad::TempStorage temp_storage;
    // Load a segment of consecutive items that are blocked across threads
    int thread_data[ItemsPerThread];
    int offset = threadIdx.x * ItemsPerThread;
    BlockLoad(temp_storage).Load(d_data + offset, thread_data);

    // Print loaded data
    printf("Thread %d loaded: %d %d %d %d\n", threadIdx.x, thread_data[0], thread_data[1], thread_data[2], thread_data[3]);
}

int main()
{
    int h_data[N];
    init_data(h_data, N);
    int *d_data;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    //CHECK:  q_ct1.parallel_for(
    //CHECK-NEXT:        sycl::nd_range<3>(GridSize * BlockSize, BlockSize),
    //CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
    //CHECK-NEXT:          BlockLoadKernel(dev_data, item_ct1);
    //CHECK-NEXT:        });

    dim3 block(BlockSize);
    dim3 grid((N + BlockSize - 1) / BlockSize);

    BlockLoadKernel<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();
    //verify_data(d_data, N);   

    cudaFree(d_data);

    return 0;
}