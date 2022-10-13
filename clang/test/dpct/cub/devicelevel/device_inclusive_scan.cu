// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_inclusive_scan %S/device_inclusive_scan.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_inclusive_scan/device_inclusive_scan.dp.cpp --match-full-lines %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>

#define N 10

struct CustomSum {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

// CHECK: DPCT1026:{{.*}}
// CHECK: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + N, device_out, op);
void test_1() {
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int host_out[10];
  CustomSum op;
  cudaMalloc((void **)&device_in, N * sizeof(int));
  cudaMalloc((void **)&device_out, N * sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cub::DeviceScan::InclusiveScan(device_tmp, n_device_tmp, device_in, device_out, op, N);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceScan::InclusiveScan((void *)device_tmp, n_device_tmp, device_in, device_out, op, N);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
}

// CHECK: DPCT1027:{{.*}}
// CHECK: 0, 0;
// CHECK: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + N, device_out, op);
void test_2() {
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int host_out[10];
  CustomSum op;
  cudaMalloc((void **)&device_in, N * sizeof(int));
  cudaMalloc((void **)&device_out, N * sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cub::DeviceScan::InclusiveScan(device_tmp, n_device_tmp, device_in, device_out, op, N), 0;
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceScan::InclusiveScan((void *)device_tmp, n_device_tmp, device_in, device_out, op, N);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
}

// CHECK: dpct::queue_ptr stream = (dpct::queue_ptr)(void *)(uintptr_t)5;
// CHECK: DPCT1026:{{.*}}
// CHECK: oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy(*stream), device_in, device_in + N, device_out, op);
void test_3() {
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int host_out[10];
  CustomSum op;
  cudaMalloc((void **)&device_in, N * sizeof(int));
  cudaMalloc((void **)&device_out, N * sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cudaStream_t stream = (cudaStream_t)(void *)(uintptr_t)5;
  cub::DeviceScan::InclusiveScan(device_tmp, n_device_tmp, device_in, device_out, op, N, stream);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceScan::InclusiveScan((void *)device_tmp, n_device_tmp, device_in, device_out, op, N, stream);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
}
