// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_reduce %S/device_reduce.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_reduce/device_reduce.dp.cpp %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cub/cub.cuh>
#include <iostream>

struct CustomMin {
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

// CHECK: DPCT1026:{{.*}}
// CHECK: q_ct1.fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + n, 0, op), 1).wait();
void test_1() {
  int n = 7;
  size_t n_tmp;
  CustomMin op;
  int *d_in, *d_out, *tmp = nullptr;
  int in[] = {8, 6, 7, 5, -1, 0, 9};
  cudaMalloc((void **)&d_in, sizeof(in));
  cudaMalloc((void **)&d_out, sizeof(in));
  cudaMemcpy((void *)d_in, (void *)in, sizeof(in), cudaMemcpyHostToDevice);
  cub::DeviceReduce::Reduce(tmp, n_tmp, d_in, d_out, n, op, 0);
  cudaMalloc((void **)&tmp, n_tmp);
  cub::DeviceReduce::Reduce(tmp, n_tmp, d_in, d_out, n, op, 0);
  cudaMemcpy((void *)in, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", in[0]);
}

// CHECK: DPCT1026:{{.*}}
// CHECK: 0, 0;
// CHECK: q_ct1.fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_in + n, 0, op), 1).wait();
void test_2() {
  int n = 7;
  size_t n_tmp;
  CustomMin op;
  int *d_in, *d_out, *tmp = nullptr;
  int in[] = {8, 6, 7, 5, -1, 0, 9};
  cudaMalloc((void **)&d_in, sizeof(in));
  cudaMalloc((void **)&d_out, sizeof(in));
  cudaMemcpy((void *)d_in, (void *)in, sizeof(in), cudaMemcpyHostToDevice);
  cub::DeviceReduce::Reduce(tmp, n_tmp, d_in, d_out, n, op, 0);
  cudaMalloc((void **)&tmp, n_tmp);
  cub::DeviceReduce::Reduce(tmp, n_tmp, d_in, d_out, n, op, 0);
  cudaMemcpy((void *)in, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", in[0]);
}

// CHECK: dpct::queue_ptr stream = (dpct::queue_ptr)(void *)(uintptr_t)5;
// CHECK: DPCT1026:{{.*}}
// CHECK: stream->fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + n, 0, op), 1).wait();
void test_3() {
  int n = 7;
  size_t n_tmp;
  CustomMin op;
  int *d_in, *d_out, *tmp = nullptr;
  int in[] = {8, 6, 7, 5, -1, 0, 9};
  cudaMalloc((void **)&d_in, sizeof(in));
  cudaMalloc((void **)&d_out, sizeof(in));
  cudaMemcpy((void *)d_in, (void *)in, sizeof(in), cudaMemcpyHostToDevice);
  cudaStream_t stream = (cudaStream_t)(void *)(uintptr_t)5;
  cub::DeviceReduce::Reduce(tmp, n_tmp, d_in, d_out, n, op, 0, stream);
  cudaMalloc((void **)&tmp, n_tmp);
  cub::DeviceReduce::Reduce(tmp, n_tmp, d_in, d_out, n, op, 0, stream);
  cudaMemcpy((void *)in, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", in[0]);
}
