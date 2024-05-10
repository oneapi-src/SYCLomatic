// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/blocklevel/blockreduce_p3 %S/blockreduce_p3.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/blocklevel/blockreduce_p3/blockreduce_p3.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/blocklevel/blockreduce_p3/blockreduce_p3.dp.cpp -o %T/blocklevel/blockreduce_p3/blockreduce_p3.dp.o %}

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

template <typename T>
__device__ __forceinline__ float reduce_topk_op_2(const float &a,
                                                  const float &b) {
  return a > b ? a : b;
}

__global__ void reduce_kernel(float *da) {
  typedef cub::BlockReduce<float, 32> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int id = threadIdx.x;
  BlockReduce rd(temp_storage);
  float temp = rd.Reduce(da[id], reduce_topk_op_2<float>);
  if (id == 0) {
    da[id] = temp;
  }
  __syncthreads();
}

__global__ void reduce_kernel1(float *da) {
  typedef cub::BlockReduce<float, 32> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int id = threadIdx.x;
  float temp = BlockReduce(temp_storage).Reduce(da[id], reduce_topk_op_2<float>);
  if (id == 0) {
    da[id] = temp;
  }
  __syncthreads();
}

template <class T, int THREAD_PRE_BLOCK>
__global__ void reduce_kernel_dependent(T *da) {
  typedef cub::BlockReduce<T, THREAD_PRE_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int id = threadIdx.x;
  BlockReduce rd(temp_storage);
  float temp = rd.Reduce(da[id], reduce_topk_op_2<float>);
  if (id == 0) {
    da[id] = temp;
  }
  __syncthreads();
}

template <class T, int THREAD_PRE_BLOCK>
__global__ void reduce_kernel_dependent1(T *da) {
  typedef cub::BlockReduce<T, THREAD_PRE_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int id = threadIdx.x;
  float temp = BlockReduce(temp_storage).Reduce(da[id], reduce_topk_op_2<T>);
  if (id == 0) {
    da[id] = temp;
  }
  __syncthreads();
}

int main() {
  int N = 32;
  {
    float *ha = (float *)malloc(N * sizeof(float));
    float *da;
    cudaMalloc(&da, N * sizeof(float));

    for (int i = 0; i < N; i++) {
      ha[i] = i * 1.0f;
    }

    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    reduce_kernel<<<1, 32>>>(da);
    reduce_kernel1<<<1, 32>>>(da);
    reduce_kernel_dependent<float, 32><<<1, 32>>>(da);
    reduce_kernel_dependent1<float, 32><<<1, 32>>>(da);
    cudaMemcpy(ha, da, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << ha[0] << std::endl;
    cudaFree(da);
    free(ha);
  }

  {
    int *ha = (int *)malloc(N * sizeof(int));
    int *da;
    cudaMalloc(&da, N * sizeof(int));

    for (int i = 0; i < N; i++) {
      ha[i] = i * 1.0f;
    }

    cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);
    reduce_kernel_dependent<int, 32><<<1, 32>>>(da);
    reduce_kernel_dependent1<int, 32><<<1, 32>>>(da);
    cudaMemcpy(ha, da, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << ha[0] << std::endl;
    cudaFree(da);
    free(ha);
  }


  return 0;
}
