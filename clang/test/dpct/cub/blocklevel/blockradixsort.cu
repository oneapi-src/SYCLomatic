// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/blocklevel/blockradixsort %S/blockradixsort.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockradixsort.dp.cpp --match-full-lines %s

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32

const int N = 32;
const int BlockSize = 32;
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

//CHECK: void BlockRadixSortKernel(int* d_data,
//CHECK:  const sycl::nd_item<3> &item_ct1) {
//CHECK:  dpct::group::radix_sort radixsort_obj;
//CHECK:  int threadid = item_ct1.get_local_id(2);
//CHECK:  d_data[threadid*ItemsPerThread] = (threadid*2)%32;
//CHECK:  radixsort_obj.sort(item_ct1.get_group(), d_data);
//CHECK:}

__global__ void BlockRadixSortKernel(int *d_data)
{
    typedef cub::BlockRadixSort<int, BlockSize, ItemsPerThread> BlockRadixSortT;
    __shared__ typename BlockRadixSortT::TempStorage sort_temp_storage;
    int threadid = threadIdx.x;
    d_data[threadid*ItemsPerThread] = (threadid*2)%32;    
    BlockRadixSortT(sort_temp_storage).Sort(d_data); 
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
