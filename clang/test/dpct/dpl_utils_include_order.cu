// RUN: dpct --format-range=none -out-root %T/dpl_utils_include_order %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/dpl_utils_include_order/dpl_utils_include_order.dp.cpp %s

// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <iostream>
// CHECK-NEXT: #include <vector>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define DATA_NUM 100

template<typename T = int>
void init_data(T* data, int num) {
  T host_data[DATA_NUM];
  for(int i = 0; i < num; i++)
    host_data[i] = i;
  cudaMemcpy(data, host_data, num * sizeof(T), cudaMemcpyHostToDevice);
}
template<typename T = int>
bool verify_data(T* data, T* expect, int num, int step = 1) {
  T host_data[DATA_NUM];
  cudaMemcpy(host_data, data, num * sizeof(T), cudaMemcpyDeviceToHost);
  for(int i = 0; i < num; i = i + step) {
    if(host_data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}
template<typename T = int>
void print_data(T* data, int num, bool IsHost = false) {
  if(IsHost) {
    for (int i = 0; i < num; i++) {
      std::cout << data[i] << ", ";
      if((i+1)%32 == 0)
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
  }
  T host_data[DATA_NUM];
  cudaMemcpy(host_data, data, num * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < num; i++) {
    std::cout << host_data[i] << ", ";
    if((i+1)%32 == 0)
        std::cout << std::endl;
  }
  std::cout << std::endl;
}

// cub::DeviceSelect::Flagged
bool test_device_select_flagged() {
  static const int n = 10;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_flagged = nullptr;
  int *device_select_num = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int host_flagged[n] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int expect_out[] = {1, 3, 5, 7, 9};
  int expect_select_num = 5;
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMalloc((void **)&device_flagged, n * sizeof(int));
  cudaMalloc((void **)&device_select_num, sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_flagged, host_flagged, sizeof(host_flagged),
             cudaMemcpyHostToDevice);
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in,
                             device_flagged, device_out, device_select_num,
                             n);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in,
                             device_flagged, device_out, device_select_num,
                             n);
  cudaDeviceSynchronize();

  if (!verify_data(device_select_num, &expect_select_num, 1)) {
    std::cout << "cub::DeviceSelect::Flagged select_num verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(&expect_select_num, 1, true);
    std::cout << "current result:\n";
    print_data<int>(device_select_num, 1);
    return false;
  }

  if (!verify_data(device_out, (int *)expect_out, expect_select_num)) {
    std::cout << "cub::DeviceSelect::Flagged output data verify failed\n";
    std::cout << "expect:\n";
    print_data<int>(expect_out, 1, true);
    std::cout << "current result:\n";
    print_data<int>(device_out, 1);
    return false;
  }
  return true;
}

int main() {
  if (test_device_select_flagged())
    std::cout << "cub::DeviceSelect::Flagged Pass\n";
  return 0;
}
