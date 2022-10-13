// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// UNSUPPORTED: windows
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/redundent_call_with_gnu_ext %S/redundent_call_with_gnu_ext.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/redundent_call_with_gnu_ext/redundent_call_with_gnu_ext.dp.cpp --match-full-lines %s

// CHECK:#include <oneapi/dpl/execution>
// CHECK:#include <oneapi/dpl/algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <cassert>

// CHECK:DPCT1026:{{.*}}
// CHECK:DPCT1026:{{.*}}
// CHECK:DPCT1026:{{.*}}
// CHECK:DPCT1026:{{.*}}
void test_1() {
  int n = 10;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int host_out[10];
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cub::DeviceReduce::Sum(0, n_device_tmp, device_in, device_out, n);
  cub::DeviceReduce::Sum(NULL, n_device_tmp, device_in, device_out, n);
  cub::DeviceReduce::Sum(__null, n_device_tmp, device_in, device_out, n);
  cub::DeviceReduce::Sum(nullptr, n_device_tmp, device_in, device_out, n);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceReduce::Sum((void *)device_tmp, n_device_tmp, device_in, device_out, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
  assert(host_out[0] == 55);
}
