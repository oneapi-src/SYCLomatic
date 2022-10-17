// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/device_select_flagged %S/device_select_flagged.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/device_select_flagged/device_select_flagged.dp.cpp --match-full-lines %s

// CHECK:#include <dpct/dpl_utils.hpp>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>

// CHECK: DPCT1026:{{.*}}
// CHECK: q_ct1.fill(device_select_num, std::distance(device_out, dpct::copy_if(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + n, device_flagged, device_out, [](const auto &t) -> bool { return t; })), 1).wait();
void test_1() {
   int n = 5;
  int num = 0;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_flagged = nullptr;
  int *device_select_num = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] =       {1, 2, 3, 4, 5};
  int host_flagged[] =  {0, 1, 0, 1, 0};
  int host_out[5];
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMalloc((void **)&device_flagged, n * sizeof(int));
  cudaMalloc((void **)&device_select_num, sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cudaMemcpy(device_flagged, host_flagged, sizeof(host_flagged), cudaMemcpyHostToDevice);
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in, device_flagged, device_out, device_select_num, n);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in, device_flagged, device_out, device_select_num, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)&num, (void *)device_select_num, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
  cudaFree(device_flagged);
  cudaFree(device_select_num);
  for (int i = 0; i < num; ++i) {
    printf("%d\n", host_out[i]);
  }
}

// CHECK: DPCT1027:{{.*}}
// CHECK: 0, 0;
// CHECK: q_ct1.fill(device_select_num, std::distance(device_out, dpct::copy_if(oneapi::dpl::execution::device_policy(q_ct1), device_in, device_in + n, device_flagged, device_out, [](const auto &t) -> bool { return t; })), 1).wait();
void test_2() {
  int n = 5;
  int num = 0;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_flagged = nullptr;
  int *device_select_num = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] =       {1, 2, 3, 4, 5};
  int host_flagged[] =  {0, 1, 0, 1, 0};
  int host_out[5];
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMalloc((void **)&device_flagged, n * sizeof(int));
  cudaMalloc((void **)&device_select_num, sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cudaMemcpy(device_flagged, host_flagged, sizeof(host_flagged), cudaMemcpyHostToDevice);
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in, device_flagged, device_out, device_select_num, n), 0;
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in, device_flagged, device_out, device_select_num, n);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)&num, (void *)device_select_num, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
  cudaFree(device_flagged);
  cudaFree(device_select_num);
  for (int i = 0; i < num; ++i) {
    printf("%d\n", host_out[i]);
  }
}

// CHECK: dpct::queue_ptr stream = (dpct::queue_ptr)(void *)(uintptr_t)5;
// CHECK: DPCT1026:{{.*}}
// CHECK: stream->fill(device_select_num, std::distance(device_out, dpct::copy_if(oneapi::dpl::execution::device_policy(*stream), device_in, device_in + n, device_flagged, device_out, [](const auto &t) -> bool { return t; })), 1).wait();
void test_3() {
   int n = 5;
  int num = 0;
  int *device_in = nullptr;
  int *device_out = nullptr;
  int *device_flagged = nullptr;
  int *device_select_num = nullptr;
  int *device_tmp = nullptr;
  size_t n_device_tmp = 0;
  int host_in[] =       {1, 2, 3, 4, 5};
  int host_flagged[] =  {0, 1, 0, 1, 0};
  int host_out[5];
  cudaMalloc((void **)&device_in, n * sizeof(int));
  cudaMalloc((void **)&device_out, n * sizeof(int));
  cudaMalloc((void **)&device_flagged, n * sizeof(int));
  cudaMalloc((void **)&device_select_num, sizeof(int));
  cudaMemcpy(device_in, (void *)host_in, sizeof(host_in), cudaMemcpyHostToDevice);
  cudaMemcpy(device_flagged, host_flagged, sizeof(host_flagged), cudaMemcpyHostToDevice);
  cudaStream_t stream = (cudaStream_t)(void *)(uintptr_t)5;
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in, device_flagged, device_out, device_select_num, n, stream);
  cudaMalloc((void **)&device_tmp, n_device_tmp);
  cub::DeviceSelect::Flagged(device_tmp, n_device_tmp, device_in, device_flagged, device_out, device_select_num, n, stream);
  cudaMemcpy((void *)host_out, (void *)device_out, sizeof(host_out), cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)&num, (void *)device_select_num, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_tmp);
  cudaFree(device_flagged);
  cudaFree(device_select_num);
  for (int i = 0; i < num; ++i) {
    printf("%d\n", host_out[i]);
  }
}
