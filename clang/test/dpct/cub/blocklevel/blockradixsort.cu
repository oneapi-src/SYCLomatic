// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/blocklevel/blockradixsort %S/blockradixsort.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockradixsort.dp.cpp --match-full-lines %s

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32

const int N = 128;
const int BlockSize = 128;
const int ItemsPerThread = 1;


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

//CHECK: void BlockRadixSortKernel(int* data,
//CHECK-NEXT:  const sycl::nd_item<3> &item_ct1) {
//CHECK-EMPTY:
//CHECK-NEXT:  int threadid = item_ct1.get_local_id(2);
//CHECK-EMPTY:
//CHECK-NEXT:  int input = data[threadid];
//CHECK-NEXT:  dpct::group::radix_sort::sort(item_ct1, input, 0 , 8*sizeof(input));
//CHECK-NEXT:  
//CHECK-NEXT:}

__global__ void BlockRadixSortKernel(int *d_data)
{
    typedef cub::BlockRadixSort<int, BlockSize, ItemsPerThread> BlockRadixSortT;
    __shared__ typename BlockRadixSortT::TempStorage sort_temp_storage;
    
    int thread_data[ItemsPerThread];
    int block_offset = blockIdx.x * BlockSize * ItemsPerThread;
    
    typedef cub::BlockLoad<int, BlockSize, ItemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    __shared__ typename BlockLoadT::TempStorage load_temp_storage;
    BlockLoadT(load_temp_storage).Load(d_data + block_offset, thread_data);
    
   __syncthreads();
   
   BlockRadixSortT(sort_temp_storage).SortKeys(thread_data);
   
   typedef cub::BlockStore<int, BlockSize, ItemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
   __shared__ typename BlockStoreT::TempStorage store_temp_storage;
   BlockStoreT(store_temp_storage).Store(d_data + block_offset, thread_data);
   
   __syncthreads();
    
 
}

int main()
{
    int h_data[N];
    init_data(h_data, N);
    int *d_data;
    cudaMalloc((void**)&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    //CHECK:  q_ct1.parallel_for(
    //CHECK-NEXT:        sycl::nd_range<3>(1, BlockSize),
    //CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
    //CHECK-NEXT:          BlockRadixSortKernel(d_data, item_ct1);
    //CHECK-NEXT:        });

    BlockRadixSortKernel<<<1, BlockSize>>>(d_data);
    
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    print_data(h_data, N);  

    cudaFree(d_data);

    return 0;
}