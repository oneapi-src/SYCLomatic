// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=restricted -out-root %T/USM-restricted %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/USM-restricted/USM-restricted.dp.cpp %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <memory>

#define MY_SAFE_CALL(CALL) do {    \
  int Error = CALL;                \
} while (0)

__constant__ float constData[1234567 * 4];

int foo_b(int a){
  return 0;
}

void foo() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  int errorCode;

  cudaPitchedPtr p_A;
  cudaExtent e;
  cudaMemcpy3DParms parms;
  cudaStream_t stream;

  /// malloc
  // CHECK: d_A = (float *)sycl::malloc_device(size, q_ct1);
  cudaMalloc((void **)&d_A, size);
  // CHECK: errorCode = (d_A = (float *)sycl::malloc_device(size, q_ct1), 0);
  errorCode = cudaMalloc((void **)&d_A, size);
  // CHECK: MY_SAFE_CALL((d_A = (float *)sycl::malloc_device(size, q_ct1), 0));
  MY_SAFE_CALL(cudaMalloc((void **)&d_A, size));

  // CHECK: d_A = (float *)sycl::malloc_device(sizeof(sycl::double2) + size, q_ct1);
  // CHECK-NEXT: d_A = (float *)sycl::malloc_device(sizeof(sycl::uchar4) + size, q_ct1);
  // CHECK-NEXT: d_A = (float *)sycl::malloc_device(sizeof(d_A[0]), q_ct1);
  cudaMalloc((void **)&d_A, sizeof(double2) + size);
  cudaMalloc((void **)&d_A, sizeof(uchar4) + size);
  cudaMalloc((void **)&d_A, sizeof(d_A[0]));
  
  // CHECK: d_A = (float *)dpct::dpct_malloc(size, size, size);
  cudaMallocPitch((void **)&d_A, &size, size, size);
  // CHECK: p_A = dpct::dpct_malloc(e);
  cudaMalloc3D(&p_A, e);

  // CHECK: h_A = (float *)sycl::malloc_host(size, q_ct1);
  cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
  // CHECK: errorCode = (h_A = (float *)sycl::malloc_host(size, q_ct1), 0);
  errorCode = cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
  // CHECK: MY_SAFE_CALL((h_A = (float *)sycl::malloc_host(size, q_ct1), 0));
  MY_SAFE_CALL(cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault));

  // CHECK: /*
  // CHECK-NEXT: DPCT1048:{{[0-9]+}}: The original value cudaHostAllocDefault is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_A = (float *)sycl::malloc_host(sizeof(sycl::double2) - size, q_ct1);
  cudaHostAlloc((void **)&h_A, sizeof(double2) - size, cudaHostAllocDefault);
  // CHECK: /*
  // CHECK-NEXT: DPCT1048:{{[0-9]+}}: The original value cudaHostAllocDefault is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: h_A = (float *)sycl::malloc_host(sizeof(sycl::uchar4) - size, q_ct1);
  cudaHostAlloc((void **)&h_A, sizeof(uchar4) - size, cudaHostAllocDefault);

  // CHECK: h_A = (float *)sycl::malloc_host(size, q_ct1);
  cudaMallocHost((void **)&h_A, size);
  // CHECK: errorCode = (h_A = (float *)sycl::malloc_host(size, q_ct1), 0);
  errorCode = cudaMallocHost((void **)&h_A, size);
  // CHECK: MY_SAFE_CALL((h_A = (float *)sycl::malloc_host(size, q_ct1), 0));
  MY_SAFE_CALL(cudaMallocHost((void **)&h_A, size));

  // CHECK: h_A = (float *)sycl::malloc_host(size, q_ct1);
  cuMemAllocHost((void **)&h_A, size);
  // CHECK: errorCode = (h_A = (float *)sycl::malloc_host(size, q_ct1), 0);
  errorCode = cuMemAllocHost((void **)&h_A, size);
  // CHECK: MY_SAFE_CALL((h_A = (float *)sycl::malloc_host(size, q_ct1), 0));
  MY_SAFE_CALL(cuMemAllocHost((void **)&h_A, size));

  // CHECK: h_A = (float *)sycl::malloc_host(sizeof(sycl::double2) * size, q_ct1);
  // CHECK-NEXT: h_A = (float *)sycl::malloc_host(sizeof(sycl::uchar4) * size, q_ct1);
  cudaMallocHost((void **)&h_A, sizeof(double2) * size);
  cudaMallocHost((void **)&h_A, sizeof(uchar4) * size);

  // CHECK: h_A = (float *)sycl::malloc_host(size, q_ct1);
  cudaMallocHost(&h_A, size);
  // CHECK: errorCode = (h_A = (float *)sycl::malloc_host(size, q_ct1), 0);
  errorCode = cudaMallocHost(&h_A, size);
  // CHECK: MY_SAFE_CALL((h_A = (float *)sycl::malloc_host(size, q_ct1), 0));
  MY_SAFE_CALL(cudaMallocHost(&h_A, size));

  // CHECK: h_A = (float *)sycl::malloc_host(sizeof(sycl::double2) / size, q_ct1);
  // CHECK-NEXT: h_A = (float *)sycl::malloc_host(sizeof(sycl::uchar4) / size, q_ct1);
  cudaMallocHost(&h_A, sizeof(double2) / size);
  cudaMallocHost(&h_A, sizeof(uchar4) / size);

  float* buffer[2];
#define SIZE_1 (128 * 1024 * 1024)
  // CHECK: *buffer = sycl::malloc_host<float>(SIZE_1, q_ct1);
  // CHECK-NEXT: *(buffer + 1) = sycl::malloc_host<float>(SIZE_1, q_ct1);
  cudaMallocHost((void**)buffer, SIZE_1 * sizeof(float));
  cudaMallocHost((void**)(buffer + 1), SIZE_1 * sizeof(float));
#undef SIZE_1

  // CHECK: d_A = (float *)sycl::malloc_shared(size, q_ct1);
  cudaMallocManaged((void **)&d_A, size);
  // CHECK: errorCode = (d_A = (float *)sycl::malloc_shared(size, q_ct1), 0);
  errorCode = cudaMallocManaged((void **)&d_A, size);
  // CHECK: MY_SAFE_CALL((d_A = (float *)sycl::malloc_shared(size, q_ct1), 0));
  MY_SAFE_CALL(cudaMallocManaged((void **)&d_A, size));

  // CHECK: d_A = (float *)sycl::malloc_shared(sizeof(sycl::double2) + size + sizeof(sycl::uchar4), q_ct1);
  // CHECK-NEXT: d_A = (float *)sycl::malloc_shared(sizeof(sycl::double2) * size * sizeof(sycl::uchar4), q_ct1);
  cudaMallocManaged((void **)&d_A, sizeof(double2) + size + sizeof(uchar4));
  cudaMallocManaged((void **)&d_A, sizeof(double2) * size * sizeof(uchar4));

  CUdeviceptr* D_ptr;
  // CHECK: *D_ptr = (void *)sycl::malloc_shared(size, q_ct1);
  cuMemAllocManaged(D_ptr, size, CU_MEM_ATTACH_HOST);

  /// memcpy

  // CHECK: q_ct1.memcpy(d_A, h_A, size).wait();
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: errorCode  = (q_ct1.memcpy(d_A, h_A, size), 0);
  errorCode  = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(d_A, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
#define MACRO_A(x) size
#define MACRO_A2(x) MACRO_A(x)
#define MACRO_B size
#define MACOR_C(x) cudaMemcpyDeviceToHost
#define MY_SAFE_CALL2(x) MY_SAFE_CALL(x)
  //CHECK: /*
  //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: MY_SAFE_CALL2((q_ct1.memcpy(d_A, h_A, size), 0));
  MY_SAFE_CALL2(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  //CHECK: /*
  //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: MY_SAFE_CALL2((q_ct1.memcpy(d_A, h_A, MACRO_B), 0));
  MY_SAFE_CALL2(cudaMemcpy(d_A, h_A, MACRO_B, cudaMemcpyDeviceToHost));
  //CHECK: /*
  //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: MY_SAFE_CALL2((q_ct1.memcpy(d_A, h_A, MACRO_A2(1)), 0));
  MY_SAFE_CALL2(cudaMemcpy(d_A, h_A, MACRO_A2(1), MACOR_C(1)));
  //CHECK: /*
  //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: MY_SAFE_CALL2((q_ct1.memcpy(d_A, h_A, foo_b(1)).wait(), 0));
  MY_SAFE_CALL2(cudaMemcpy(d_A, h_A, foo_b(1), MACOR_C(1)));

#define SIZE 100
  // CHECK: q_ct1.memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );

  /// memcpy async

  // CHECK: q_ct1.memcpy(d_A, h_A, size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (q_ct1.memcpy(d_A, h_A, size), 0);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(d_A, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice));

  // CHECK: q_ct1.memcpy(d_A, h_A, size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: errorCode = (q_ct1.memcpy(d_A, h_A, size), 0);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(d_A, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0));

  // CHECK: stream->memcpy(d_A, h_A, size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
  // CHECK: errorCode = (stream->memcpy(d_A, h_A, size), 0);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
  // CHECK: MY_SAFE_CALL((stream->memcpy(d_A, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));

  // CHECK: dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::dpct_memcpy(h_A, size, d_A, size, size, size, dpct::device_to_host);
  cudaMemcpy2D(h_A, size, d_A, size, size, size, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  cudaMemcpy3D(&parms);

  struct cudaMemcpy3DParms *parms_pointer;
  // Followed call can't be processed.
  cudaMemcpy3D(parms_pointer);
  // CHECK: dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device);
  cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device);
  cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device, *stream);
  cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, stream);

  // CHECK: dpct::async_dpct_memcpy(h_A, size, d_A, size, size, size, dpct::device_to_host);
  cudaMemcpy2DAsync(h_A, size, d_A, size, size, size, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, size, d_A, size, size, size, dpct::device_to_host);
  cudaMemcpy2DAsync(h_A, size, d_A, size, size, size, cudaMemcpyDeviceToHost, 0);
  // CHECK: dpct::async_dpct_memcpy(h_A, size, d_A, size, size, size, dpct::device_to_host, *stream);
  cudaMemcpy2DAsync(h_A, size, d_A, size, size, size, cudaMemcpyDeviceToHost, stream);

  // CHECK: dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  cudaMemcpy3DAsync(&parms);
  // CHECK: dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  cudaMemcpy3DAsync(&parms, 0);
  // CHECK: dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1, *stream);
  cudaMemcpy3DAsync(&parms, stream);
  /// memcpy from symbol

  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  cudaMemcpyFromSymbol(h_A, "constData", size, 1);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1));

  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  cudaMemcpyFromSymbol(h_A, "constData", size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait(), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  /// memcpy from symbol async

  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 2, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 2, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 2, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0));

  // CHECK: stream->memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: stream->memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: errorCode = (stream->memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: MY_SAFE_CALL((stream->memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream));

  /// memcpy to symbol
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  cudaMemcpyToSymbol("constData", h_A, size, 1);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1));

  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  cudaMemcpyToSymbol("constData", h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait(), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  /// memcpy to symbol async

  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 2, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 2, h_A, size);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 2, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 2, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0));

  // CHECK: stream->memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: stream->memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: errorCode = (stream->memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: MY_SAFE_CALL((stream->memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream));

  /// memset

  // CHECK: q_ct1.memset(d_A, 23, size).wait();
  cudaMemset(d_A, 23, size);
  // CHECK: errorCode = (q_ct1.memset(d_A, 23, size).wait(), 0);
  errorCode = cudaMemset(d_A, 23, size);
  // CHECK: MY_SAFE_CALL((q_ct1.memset(d_A, 23, size).wait(), 0));
  MY_SAFE_CALL(cudaMemset(d_A, 23, size));

  /// memset async

  // CHECK: q_ct1.memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size);
  // CHECK: errorCode = (q_ct1.memset(d_A, 23, size), 0);
  errorCode = cudaMemsetAsync(d_A, 23, size);
  // CHECK: MY_SAFE_CALL((q_ct1.memset(d_A, 23, size), 0));
  MY_SAFE_CALL(cudaMemsetAsync(d_A, 23, size));

  // CHECK: q_ct1.memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size, 0);
  // CHECK: errorCode = (q_ct1.memset(d_A, 23, size), 0);
  errorCode = cudaMemsetAsync(d_A, 23, size, 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memset(d_A, 23, size), 0));
  MY_SAFE_CALL(cudaMemsetAsync(d_A, 23, size, 0));

  // CHECK: stream->memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size, stream);
  // CHECK: errorCode = (stream->memset(d_A, 23, size), 0);
  errorCode = cudaMemsetAsync(d_A, 23, size, stream);
  // CHECK: MY_SAFE_CALL((stream->memset(d_A, 23, size), 0));
  MY_SAFE_CALL(cudaMemsetAsync(d_A, 23, size, stream));
  
  // CHECK: dpct::dpct_memset(d_A, size, 0xf, size, size);
  cudaMemset2D(d_A, size, 0xf, size, size);
  // CHECK: dpct::dpct_memset(p_A, 0xf, e);
  cudaMemset3D(p_A, 0xf, e);

  // CHECK: dpct::async_dpct_memset(d_A, size, 0xf, size, size);
  cudaMemset2DAsync(d_A, size, 0xf, size, size);
  // CHECK: dpct::async_dpct_memset(d_A, size, 0xf, size, size);
  cudaMemset2DAsync(d_A, size, 0xf, size, size, 0);
  // CHECK: dpct::async_dpct_memset(d_A, size, 0xf, size, size, *stream);
  cudaMemset2DAsync(d_A, size, 0xf, size, size, stream);

  // CHECK: dpct::async_dpct_memset(p_A, 0xf, e);
  cudaMemset3DAsync(p_A, 0xf, e);
  // CHECK: dpct::async_dpct_memset(p_A, 0xf, e);
  cudaMemset3DAsync(p_A, 0xf, e, 0);
  // CHECK: dpct::async_dpct_memset(p_A, 0xf, e, *stream);
  cudaMemset3DAsync(p_A, 0xf, e, stream);

  // CHECK: sycl::free(h_A, q_ct1);
  cudaFreeHost(h_A);
  // CHECK: errorCode = (sycl::free(h_A, q_ct1), 0);
  errorCode = cudaFreeHost(h_A);
  // CHECK: MY_SAFE_CALL((sycl::free(h_A, q_ct1), 0));
  MY_SAFE_CALL(cudaFreeHost(h_A));

  // CHECK: d_A = h_A;
  cudaHostGetDevicePointer(&d_A, h_A, 0);
  // CHECK: errorCode = (d_A = h_A, 0);
  errorCode = cudaHostGetDevicePointer(&d_A, h_A, 0);
  // CHECK: MY_SAFE_CALL((d_A = h_A, 0));
  MY_SAFE_CALL(cudaHostGetDevicePointer(&d_A, h_A, 0));

  // CHECK: *D_ptr = h_A;
  cuMemHostGetDevicePointer(D_ptr, h_A, 0);
  // CHECK: errorCode = (*D_ptr = h_A, 0);
  errorCode = cuMemHostGetDevicePointer(D_ptr, h_A, 0);
  // CHECK: MY_SAFE_CALL((*D_ptr = h_A, 0));
  MY_SAFE_CALL(cuMemHostGetDevicePointer(D_ptr, h_A, 0));

  cudaHostRegister(h_A, size, 0);
  // CHECK: errorCode = 0;
  errorCode = cudaHostRegister(h_A, size, 0);
  // CHECK: MY_SAFE_CALL(0);
  MY_SAFE_CALL(cudaHostRegister(h_A, size, 0));

  cudaHostUnregister(h_A);
  // CHECK: errorCode = 0;
  errorCode = cudaHostUnregister(h_A);
  // CHECK: MY_SAFE_CALL(0);
  MY_SAFE_CALL(cudaHostUnregister(h_A));
}


template <typename T>
int foo2() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  int errorCode;

  cudaStream_t stream;
  /// memcpy from symbol

  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  cudaMemcpyFromSymbol(h_A, "constData", size, 1);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1));

  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  // CHECK: q_ct1.memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK: q_ct1.memcpy(h_A, constData.get_ptr(), size).wait();
  cudaMemcpyFromSymbol(h_A, "constData", size);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, constData.get_ptr(), size), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, constData.get_ptr(), size).wait(), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size));

  /// memcpy from symbol async

  // CHECK: q_ct1.memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size);
  // CHECK: q_ct1.memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, constData.get_ptr(), size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, constData.get_ptr(), size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size));
  
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1);
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size, 1);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 1);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 1));

  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 2, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 2, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 2, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0));

  // CHECK: stream->memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: stream->memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: errorCode = (stream->memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: MY_SAFE_CALL((stream->memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size), 0));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream));

  /// memcpy to symbol
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  cudaMemcpyToSymbol("constData", h_A, size, 1);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1));

  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  // CHECK: q_ct1.memcpy(constData.get_ptr(), h_A, size);
  cudaMemcpyToSymbol(constData, h_A, size);
  // CHECK: q_ct1.memcpy(constData.get_ptr(), h_A, size).wait();
  cudaMemcpyToSymbol("constData", h_A, size);
  // CHECK: errorCode = (q_ct1.memcpy(constData.get_ptr(), h_A, size), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(constData.get_ptr(), h_A, size).wait(), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size));

  /// memcpy to symbol async
  // CHECK: q_ct1.memcpy(constData.get_ptr(), h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size);
  // CHECK: q_ct1.memcpy(constData.get_ptr(), h_A, size);
  cudaMemcpyToSymbolAsync("constData", h_A, size);
  // CHECK: errorCode = (q_ct1.memcpy(constData.get_ptr(), h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(constData.get_ptr(), h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size));

  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1);
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 1);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1));

  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 2, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 2, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 2, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0));

  // CHECK: stream->memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: stream->memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: errorCode = (stream->memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: MY_SAFE_CALL((stream->memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size), 0));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream));
}

template int foo2<float>();
template int foo2<int>();

void foo3() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  int errorCode;
  cudaPitchedPtr p_A;
  cudaExtent e;
  cudaMemcpy3DParms parms;
  int *data;
  size_t width, height, depth, pitch, woffset, hoffset;
  cudaArray_t a1;
  int deviceID = 0;

  // CHECK: auto s1 = std::make_shared<sycl::queue *>((sycl::queue *)&q_ct1);
  // CHECK: auto s2 = std::make_shared<sycl::queue *>(&q_ct1);
  // CHECK: auto s3 = std::make_shared<sycl::queue *>(&q_ct1);
  auto s1 = std::make_shared<cudaStream_t>((cudaStream_t)cudaStreamDefault);
  auto s2 = std::make_shared<cudaStream_t>(cudaStreamLegacy);
  auto s3 = std::make_shared<cudaStream_t>(cudaStreamPerThread);

  // CHECK: q_ct1.memcpy(d_A, h_A, size);
  // CHECK: q_ct1.memcpy(d_A, h_A, size);
  // CHECK: q_ct1.memcpy(d_A, h_A, size);
  // CHECK: errorCode = (q_ct1.memcpy(d_A, h_A, size), 0);
  // CHECK: errorCode = (q_ct1.memcpy(d_A, h_A, size), 0);
  // CHECK: errorCode = (q_ct1.memcpy(d_A, h_A, size), 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(d_A, h_A, size), 0));
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(d_A, h_A, size), 0));
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(d_A, h_A, size), 0));
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamDefault);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamLegacy);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamPerThread);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamDefault);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamLegacy);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamPerThread));


  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  // CHECK: q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  // CHECK: errorCode = (q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamDefault);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamLegacy);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamPerThread);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamDefault);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamLegacy);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamPerThread));

  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 3, size);
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 3, size);
  // CHECK: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 3, size);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0));
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0));
  // CHECK: MY_SAFE_CALL((q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0));
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, cudaStreamDefault);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, cudaStreamDefault);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, cudaStreamPerThread));

  // CHECK: dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device);
  // CHECK: dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device);
  // CHECK: dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device), 0);
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device), 0));
  cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, cudaStreamDefault);
  cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, cudaStreamLegacy);
  cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, cudaStreamPerThread);
  errorCode = cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, cudaStreamDefault);
  errorCode = cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, cudaStreamLegacy);
  errorCode = cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpy2DAsync(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice, cudaStreamPerThread));

  // CHECK: dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  // CHECK: dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  // CHECK: dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0);
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0));
  cudaMemcpy3DAsync(&parms, cudaStreamDefault);
  cudaMemcpy3DAsync(&parms, cudaStreamLegacy);
  cudaMemcpy3DAsync(&parms, cudaStreamPerThread);
  errorCode = cudaMemcpy3DAsync(&parms, cudaStreamDefault);
  errorCode = cudaMemcpy3DAsync(&parms, cudaStreamLegacy);
  errorCode = cudaMemcpy3DAsync(&parms, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpy3DAsync(&parms, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpy3DAsync(&parms, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpy3DAsync(&parms, cudaStreamPerThread));


  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1));
  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1));
  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1));
  // CHECK: errorCode = (dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)), 0);
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)), 0));
  cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, cudaStreamDefault);
  cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  errorCode = cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, cudaStreamDefault);
  errorCode = cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  errorCode = cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpy2DFromArrayAsync(data, pitch, a1, woffset, hoffset, width, height, cudaMemcpyDeviceToHost, cudaStreamPerThread));


  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1));
  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1));
  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1));
  // CHECK: errorCode = (dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)), 0);
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)), 0));
  cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, cudaStreamDefault);
  cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  errorCode = cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, cudaStreamDefault);
  errorCode = cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  errorCode = cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpy2DToArrayAsync(a1, woffset, hoffset, data, pitch, width, height, cudaMemcpyDeviceToHost, cudaStreamPerThread));


  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1));
  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1));
  // CHECK: dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1));
  // CHECK: errorCode = (dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)), 0);
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)), 0));
  cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, cudaStreamDefault);
  cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  errorCode = cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, cudaStreamDefault);
  errorCode = cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  errorCode = cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpyToArrayAsync(a1, woffset, hoffset, data, width, cudaMemcpyDeviceToHost, cudaStreamPerThread));


  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1));
  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1));
  // CHECK: dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1));
  // CHECK: errorCode = (dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)), 0);
  // CHECK: errorCode = (dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)), 0);
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)), 0));
  cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamDefault);
  cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  errorCode = cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamDefault);
  errorCode = cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  errorCode = cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamPerThread));


  // CHECK: q_ct1.memset(d_A, 23, size);
  // CHECK: q_ct1.memset(d_A, 23, size);
  // CHECK: q_ct1.memset(d_A, 23, size);
  // CHECK: errorCode = (q_ct1.memset(d_A, 23, size), 0);
  // CHECK: errorCode = (q_ct1.memset(d_A, 23, size), 0);
  // CHECK: errorCode = (q_ct1.memset(d_A, 23, size), 0);
  // CHECK: MY_SAFE_CALL((q_ct1.memset(d_A, 23, size), 0));
  // CHECK: MY_SAFE_CALL((q_ct1.memset(d_A, 23, size), 0));
  // CHECK: MY_SAFE_CALL((q_ct1.memset(d_A, 23, size), 0));
  cudaMemsetAsync(d_A, 23, size, cudaStreamDefault);
  cudaMemsetAsync(d_A, 23, size, cudaStreamLegacy);
  cudaMemsetAsync(d_A, 23, size, cudaStreamPerThread);
  errorCode = cudaMemsetAsync(d_A, 23, size, cudaStreamDefault);
  errorCode = cudaMemsetAsync(d_A, 23, size, cudaStreamLegacy);
  errorCode = cudaMemsetAsync(d_A, 23, size, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemsetAsync(d_A, 23, size, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemsetAsync(d_A, 23, size, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemsetAsync(d_A, 23, size, cudaStreamPerThread));


  // CHECK: dpct::async_dpct_memset(d_A, size, 0xf, size, size);
  // CHECK: dpct::async_dpct_memset(d_A, size, 0xf, size, size);
  // CHECK: dpct::async_dpct_memset(d_A, size, 0xf, size, size);
  // CHECK: errorCode = (dpct::async_dpct_memset(d_A, size, 0xf, size, size), 0);
  // CHECK: errorCode = (dpct::async_dpct_memset(d_A, size, 0xf, size, size), 0);
  // CHECK: errorCode = (dpct::async_dpct_memset(d_A, size, 0xf, size, size), 0);
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memset(d_A, size, 0xf, size, size), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memset(d_A, size, 0xf, size, size), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memset(d_A, size, 0xf, size, size), 0));
  cudaMemset2DAsync(d_A, size, 0xf, size, size, cudaStreamDefault);
  cudaMemset2DAsync(d_A, size, 0xf, size, size, cudaStreamLegacy);
  cudaMemset2DAsync(d_A, size, 0xf, size, size, cudaStreamPerThread);
  errorCode = cudaMemset2DAsync(d_A, size, 0xf, size, size, cudaStreamDefault);
  errorCode = cudaMemset2DAsync(d_A, size, 0xf, size, size, cudaStreamLegacy);
  errorCode = cudaMemset2DAsync(d_A, size, 0xf, size, size, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemset2DAsync(d_A, size, 0xf, size, size, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemset2DAsync(d_A, size, 0xf, size, size, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemset2DAsync(d_A, size, 0xf, size, size, cudaStreamPerThread));


  // CHECK: dpct::async_dpct_memset(p_A, 0xf, e);
  // CHECK: dpct::async_dpct_memset(p_A, 0xf, e);
  // CHECK: dpct::async_dpct_memset(p_A, 0xf, e);
  // CHECK: errorCode = (dpct::async_dpct_memset(p_A, 0xf, e), 0);
  // CHECK: errorCode = (dpct::async_dpct_memset(p_A, 0xf, e), 0);
  // CHECK: errorCode = (dpct::async_dpct_memset(p_A, 0xf, e), 0);
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memset(p_A, 0xf, e), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memset(p_A, 0xf, e), 0));
  // CHECK: MY_SAFE_CALL((dpct::async_dpct_memset(p_A, 0xf, e), 0));
  cudaMemset3DAsync(p_A, 0xf, e, cudaStreamDefault);
  cudaMemset3DAsync(p_A, 0xf, e, cudaStreamLegacy);
  cudaMemset3DAsync(p_A, 0xf, e, cudaStreamPerThread);
  errorCode = cudaMemset3DAsync(p_A, 0xf, e, cudaStreamDefault);
  errorCode = cudaMemset3DAsync(p_A, 0xf, e, cudaStreamLegacy);
  errorCode = cudaMemset3DAsync(p_A, 0xf, e, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemset3DAsync(p_A, 0xf, e, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemset3DAsync(p_A, 0xf, e, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemset3DAsync(p_A, 0xf, e, cudaStreamPerThread));


  // CHECK: dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(d_A,100);
  // CHECK: dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(d_A,100);
  // CHECK: dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(d_A,100);
  // CHECK: errorCode = (dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(d_A,100), 0);
  // CHECK: errorCode = (dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(d_A,100), 0);
  // CHECK: errorCode = (dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(d_A,100), 0);
  // CHECK: MY_SAFE_CALL((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(d_A,100), 0));
  // CHECK: MY_SAFE_CALL((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(d_A,100), 0));
  // CHECK: MY_SAFE_CALL((dpct::dev_mgr::instance().get_device(deviceID).default_queue().prefetch(d_A,100), 0));
  cudaMemPrefetchAsync (d_A, 100, deviceID, cudaStreamDefault);
  cudaMemPrefetchAsync (d_A, 100, deviceID, cudaStreamLegacy);
  cudaMemPrefetchAsync (d_A, 100, deviceID, cudaStreamPerThread);
  errorCode = cudaMemPrefetchAsync (d_A, 100, deviceID, cudaStreamDefault);
  errorCode = cudaMemPrefetchAsync (d_A, 100, deviceID, cudaStreamLegacy);
  errorCode = cudaMemPrefetchAsync (d_A, 100, deviceID, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemPrefetchAsync (d_A, 100, deviceID, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemPrefetchAsync (d_A, 100, deviceID, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemPrefetchAsync (d_A, 100, deviceID, cudaStreamPerThread));
}

/// cuda driver memory api
void foo4(){
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);

  int errorCode;
  // CHECK: /*
  // CHECK: DPCT1048:{{[0-9]+}}: The original value CU_MEMHOSTALLOC_PORTABLE is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
  // CHECK: */
  // CHECK: h_A = (float *)sycl::malloc_host(size, q_ct1);
  cuMemHostAlloc((void **)&h_A, size, CU_MEMHOSTALLOC_PORTABLE);
  // CHECK: /*
  // CHECK: DPCT1048:{{[0-9]+}}: The original value CU_MEMHOSTALLOC_PORTABLE is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
  // CHECK: */
  // CHECK: errorCode = (h_A = (float *)sycl::malloc_host(size, q_ct1), 0);
  errorCode = cuMemHostAlloc((void **)&h_A, size, CU_MEMHOSTALLOC_PORTABLE);
  // CHECK: /*
  // CHECK: DPCT1048:{{[0-9]+}}: The original value CU_MEMHOSTALLOC_PORTABLE is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
  // CHECK: */
  // CHECK: MY_SAFE_CALL((h_A = (float *)sycl::malloc_host(size, q_ct1), 0));
  MY_SAFE_CALL(cuMemHostAlloc((void **)&h_A, size, CU_MEMHOSTALLOC_PORTABLE));
  // CHECK: /*
  // CHECK: DPCT1048:{{[0-9]+}}: The original value CU_MEMHOSTALLOC_PORTABLE is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
  // CHECK: */
  // CHECK: h_A = (float *)sycl::malloc_host(sizeof(sycl::double2) - size, q_ct1);
  cuMemHostAlloc((void **)&h_A, sizeof(double2) - size, CU_MEMHOSTALLOC_PORTABLE);
  // CHECK: /*
  // CHECK: DPCT1048:{{[0-9]+}}: The original value CU_MEMHOSTALLOC_PORTABLE is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
  // CHECK: */
  // CHECK: h_A = (float *)sycl::malloc_host(sizeof(sycl::uchar4) - size, q_ct1);
  cuMemHostAlloc((void **)&h_A, sizeof(uchar4) - size, CU_MEMHOSTALLOC_PORTABLE);
}

#define MY_SAFE_CALL3(CALL) {                                               \
  cudaError Error = CALL;                                                   \
  if (Error != cudaSuccess) {                                               \
    printf("%s\n", cudaGetErrorString(Error));                              \
    exit(Error);                                                            \
  }                                                                         \
}

void foo5(float* a) {
// CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16), 0));
// CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16), 0));
// CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16).wait(), 0));
  MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
  MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
  MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
}


void foo6(float* a) {
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: printf("%d\n", (q_ct1.memcpy(a, a, 16).wait(), 0));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: printf("%d\n", (q_ct1.memcpy(a, a, 16).wait(), 0));
  printf("%d\n", cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
  printf("%d\n", cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
}

__global__ void test_kernel() {}

int foo7() {
  unsigned int mem_size;
  unsigned int *h_out_data;
  unsigned int *h_data;
  unsigned int *d_out_data;
  unsigned int *d_in_data_1;
  unsigned int *d_in_data_2;
  int num_data;

  for (unsigned int i = 0; i < num_data; i++)
    h_data[i] = i;
  // CHECK: q_ct1.memcpy(d_in_data_1, h_data, mem_size).wait();
  cudaMemcpy(d_in_data_1, h_data, mem_size, cudaMemcpyHostToDevice);

  for (unsigned int i = 0; i < num_data; i++)
    h_data[i] = num_data - 1 - i;
  // CHECK: q_ct1.memcpy(d_in_data_2, h_data, mem_size);
  cudaMemcpy(d_in_data_2, h_data, mem_size, cudaMemcpyHostToDevice);

  test_kernel<<<3, 3>>>();
  cudaDeviceSynchronize();
  // CHECK: q_ct1.memcpy(h_out_data, d_out_data, mem_size).wait();
  cudaMemcpy(h_out_data, d_out_data, mem_size, cudaMemcpyDeviceToHost);

  return 0;
}

int foo8() {
  unsigned int mem_size;
  unsigned int *h_data;
  unsigned int *d_in_data_1;
  unsigned int *d_in_data_2;

  // CHECK: q_ct1.memcpy(d_in_data_1, h_data, mem_size);
  cudaMemcpy(d_in_data_1, h_data, mem_size, cudaMemcpyHostToDevice);
  // CHECK: q_ct1.memcpy(d_in_data_2, h_data, mem_size).wait();
  cudaMemcpy(d_in_data_2, h_data, mem_size, cudaMemcpyHostToDevice);
  return 0;
}

int foo9() {
  unsigned int mem_size;
  unsigned int *h_data;
  unsigned int *d_in_data_1;
  unsigned int *d_in_data_2;
  unsigned int *test = d_in_data_1;

  // CHECK: q_ct1.memcpy(d_in_data_1, h_data, mem_size).wait();
  cudaMemcpy(d_in_data_1, h_data, mem_size, cudaMemcpyHostToDevice);
  test;
  // CHECK: q_ct1.memcpy(d_in_data_2, h_data, mem_size).wait();
  cudaMemcpy(d_in_data_2, h_data, mem_size, cudaMemcpyHostToDevice);
  return 0;
}

int foo10(unsigned int *test) {
  unsigned int mem_size;
  unsigned int *data_d, *data_h;

  // CHECK: q_ct1.memcpy(data_d, data_h, mem_size).wait();
  cudaMemcpy(data_d, data_h, mem_size, cudaMemcpyHostToDevice);
  test;
  // CHECK: q_ct1.memcpy(data_d, data_h, mem_size).wait();
  cudaMemcpy(data_d, data_h, mem_size, cudaMemcpyHostToDevice);
  return 0;
}

unsigned int *global_test;

int foo11() {
  unsigned int mem_size;
  unsigned int *data_d, *data_h;

  // CHECK: q_ct1.memcpy(data_d, data_h, mem_size).wait();
  cudaMemcpy(data_d, data_h, mem_size, cudaMemcpyHostToDevice);
  global_test;
  // CHECK: q_ct1.memcpy(data_d, data_h, mem_size).wait();
  cudaMemcpy(data_d, data_h, mem_size, cudaMemcpyHostToDevice);
  return 0;
}

struct TEST {
  unsigned int t;
  void call() {
    unsigned int mem_size;
    unsigned int *data_d, *data_h;
    // CHECK: q_ct1.memcpy(data_d, data_h, mem_size);
    cudaMemcpy(data_d, data_h, mem_size, cudaMemcpyHostToDevice);
    // CHECK: q_ct1.memcpy(data_d, data_h, mem_size).wait();
    cudaMemcpy(data_d, data_h, mem_size, cudaMemcpyHostToDevice);
  }
};

int foo12() {
  TEST test;
  return 0;
}

void foo13(float* a, bool flag) {
  // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16), 0));
  // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16).wait(), 0));
  MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
  MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
  while(flag) {
    // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16), 0));
    // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16).wait(), 0));
    MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
    MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
    if(flag) {
      // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16), 0));
      // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(constData.get_ptr(), a, 16).wait(), 0));
      MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
      MY_SAFE_CALL3(cudaMemcpyToSymbol(constData, a, 16));
    } else {
      // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(constData.get_ptr(), a, 16), 0));
      // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16).wait(), 0));
      MY_SAFE_CALL3(cudaMemcpyToSymbol(constData, a, 16));
      MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
    }
    // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16), 0));
    // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16).wait(), 0));
    MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
    MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
  }

  do {
    // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16), 0));
    // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, constData.get_ptr(), 16).wait(), 0));
    MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
    MY_SAFE_CALL3(cudaMemcpyFromSymbol(a, constData, 16));
  } while(flag);

  for(;;) {
    // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, constData.get_ptr(), 16), 0));
    // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16).wait(), 0));
    MY_SAFE_CALL3(cudaMemcpyFromSymbol(a, constData, 16));
    MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
  }
  // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16), 0));
  // CHECK: MY_SAFE_CALL3((q_ct1.memcpy(a, a, 16).wait(), 0));
  MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
  MY_SAFE_CALL3(cudaMemcpy(a, a, 16, cudaMemcpyDeviceToHost));
}