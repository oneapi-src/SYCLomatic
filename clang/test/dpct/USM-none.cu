// FIXME
// UNSUPPORTED: system-windows
// RUN: dpct --format-range=none --usm-level=none -out-root %T/USM-none %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/USM-none/USM-none.dp.cpp %s

// CHECK: #define DPCT_USM_LEVEL_NONE
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <memory>

__constant__ float constData[123 * 4];

void foo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  cudaStream_t stream;

  // CHECK: d_A = (float *)dpct::dpct_malloc(size);
  cudaMalloc((void **)&d_A, size);

  /// memcpy
  // CHECK: dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  /// memcpy async
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device, *stream);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);

  /// memcpy from symbol
  // CHECK: dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbol(h_A, "constData", size, 1);
  // CHECK: dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);

  /// memcpy from symbol async
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size, 1, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 2, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size, dpct::device_to_host, *stream);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);

  /// memcpy to symbol
  // CHECK: dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbol("constData", h_A, size, 1);
  // CHECK: dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);

  /// memcpy to symbol async
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 2, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size, dpct::host_to_device, *stream);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);

  /// memset
  // CHECK: dpct::dpct_memset(d_A, 23, size);
  cudaMemset(d_A, 23, size);

  /// memset async
  // CHECK: dpct::async_dpct_memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size);
  // CHECK: dpct::async_dpct_memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size, 0);
  // CHECK: dpct::async_dpct_memset(d_A, 23, size, *stream);
  cudaMemsetAsync(d_A, 23, size, stream);

  // CHECK: h_A = (float *)malloc(size);
  cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
  // CHECK: h_A = (float *)malloc(size);
  cudaMallocHost((void **)&h_A, size);
  // CHECK: d_A = (float *)dpct::dpct_malloc(size);
  cudaMallocManaged((void **)&d_A, size);

  // CHECK: h_A = (float *)malloc(size);
  cuMemAllocHost((void **)&h_A, size);

  CUdeviceptr* D_ptr;
  // CHECK: *D_ptr = (dpct::device_ptr)dpct::dpct_malloc(size);
  cuMemAllocManaged(D_ptr, size, CU_MEM_ATTACH_HOST);

  float* buffer[2];
#define SIZE_1 (128 * 1024 * 1024)
  // CHECK: *buffer = (float *)malloc(SIZE_1 * sizeof(float));
  // CHECK-NEXT: *(buffer + 1) = (float *)malloc(SIZE_1 * sizeof(float));
  cudaMallocHost((void**)buffer, SIZE_1 * sizeof(float));
  cudaMallocHost((void**)(buffer + 1), SIZE_1 * sizeof(float));
#undef SIZE_1


  // CHECK: free(h_A);
  cudaFreeHost(h_A);

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaHostGetDevicePointer is not supported.
  // CHECK-NEXT: */
  cudaHostGetDevicePointer(&d_A, h_A, 0);

  cudaHostRegister(h_A, size, 0);
  cudaHostUnregister(h_A);
}

template <typename T>
int foo2() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  cudaStream_t stream;
  /// memcpy from symbol
  // CHECK: dpct::dpct_memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK: dpct::dpct_memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, "constData", size);
  // CHECK: dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbol(h_A, "constData", size, 1);
  // CHECK: dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);

  /// memcpy from symbol async
  // CHECK: dpct::async_dpct_memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size);
  // CHECK: dpct::async_dpct_memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 2, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 2, size);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size, 2);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 2, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size, dpct::device_to_host, *stream);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size, dpct::device_to_host, *stream);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size, 3, cudaMemcpyDeviceToHost, stream);

  /// memcpy to symbol
  // CHECK: dpct::dpct_memcpy(constData.get_ptr(), h_A, size);
  cudaMemcpyToSymbol(constData, h_A, size);
  // CHECK: dpct::dpct_memcpy(constData.get_ptr(), h_A, size);
  cudaMemcpyToSymbol("constData", h_A, size);
  // CHECK: dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbol("constData", h_A, size, 1);
  // CHECK: dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);

  /// memcpy to symbol async
  // CHECK: dpct::async_dpct_memcpy(constData.get_ptr(), h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size);
  // CHECK: dpct::async_dpct_memcpy(constData.get_ptr(), h_A, size);
  cudaMemcpyToSymbolAsync("constData", h_A, size);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 2, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 2, h_A, size);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 2);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 2, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size, dpct::host_to_device, *stream);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size, dpct::host_to_device, *stream);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 3, cudaMemcpyHostToDevice, stream);
}

template int foo2<float>();
template int foo2<int>();

#define MY_SAFE_CALL(CALL) do {    \
  int Error = CALL;                \
} while (0)

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

  // CHECK: auto s1 = std::make_shared<dpct::queue_ptr>((dpct::queue_ptr)&q_ct1);
  // CHECK: auto s2 = std::make_shared<dpct::queue_ptr>(&q_ct1);
  // CHECK: auto s3 = std::make_shared<dpct::queue_ptr>(&q_ct1);
  auto s1 = std::make_shared<cudaStream_t>((cudaStream_t)cudaStreamDefault);
  auto s2 = std::make_shared<cudaStream_t>(cudaStreamLegacy);
  auto s3 = std::make_shared<cudaStream_t>(cudaStreamPerThread);

  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device)));
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamDefault);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamLegacy);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamPerThread);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamDefault);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamLegacy);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, cudaStreamPerThread));


  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device)));
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamDefault);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamLegacy);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamPerThread);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamDefault);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamLegacy);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice, cudaStreamPerThread));

  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host);
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host)));
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
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device)));
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
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1)));
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
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1))));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1))));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, height, 1))));
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
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1))));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1))));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, pitch, pitch, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, height, 1))));
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
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1))));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1))));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), sycl::range<3>(width, 1, 1))));
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
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1))));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1))));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memcpy(dpct::pitched_data(data, width, width, 1), sycl::id<3>(0, 0, 0), a1->to_pitched_data(), sycl::id<3>(woffset, hoffset, 0), sycl::range<3>(width, 1, 1))));
  cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamDefault);
  cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  errorCode = cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamDefault);
  errorCode = cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamLegacy);
  errorCode = cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemcpyFromArrayAsync(data, a1, woffset, hoffset, width, cudaMemcpyDeviceToHost, cudaStreamPerThread));


  // CHECK: dpct::async_dpct_memset(d_A, 23, size);
  // CHECK: dpct::async_dpct_memset(d_A, 23, size);
  // CHECK: dpct::async_dpct_memset(d_A, 23, size);
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, 23, size));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, 23, size));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, 23, size));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, 23, size)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, 23, size)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, 23, size)));
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
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, size, 0xf, size, size));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, size, 0xf, size, size));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, size, 0xf, size, size));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, size, 0xf, size, size)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, size, 0xf, size, size)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memset(d_A, size, 0xf, size, size)));
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
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memset(p_A, 0xf, e));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memset(p_A, 0xf, e));
  // CHECK: errorCode = CHECK_SYCL_ERROR(dpct::async_dpct_memset(p_A, 0xf, e));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memset(p_A, 0xf, e)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memset(p_A, 0xf, e)));
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(dpct::async_dpct_memset(p_A, 0xf, e)));
  cudaMemset3DAsync(p_A, 0xf, e, cudaStreamDefault);
  cudaMemset3DAsync(p_A, 0xf, e, cudaStreamLegacy);
  cudaMemset3DAsync(p_A, 0xf, e, cudaStreamPerThread);
  errorCode = cudaMemset3DAsync(p_A, 0xf, e, cudaStreamDefault);
  errorCode = cudaMemset3DAsync(p_A, 0xf, e, cudaStreamLegacy);
  errorCode = cudaMemset3DAsync(p_A, 0xf, e, cudaStreamPerThread);
  MY_SAFE_CALL(cudaMemset3DAsync(p_A, 0xf, e, cudaStreamDefault));
  MY_SAFE_CALL(cudaMemset3DAsync(p_A, 0xf, e, cudaStreamLegacy));
  MY_SAFE_CALL(cudaMemset3DAsync(p_A, 0xf, e, cudaStreamPerThread));
}

/// cuda driver memory api
void foo4(){
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  int errorCode;
  // CHECK: h_A = (float *)malloc(size);
  cuMemHostAlloc((void **)&h_A, size, CU_MEMHOSTALLOC_PORTABLE);
  // CHECK: errorCode = CHECK_SYCL_ERROR(h_A = (float *)malloc(size));
  errorCode = cuMemHostAlloc((void **)&h_A, size, CU_MEMHOSTALLOC_PORTABLE);
  // CHECK: MY_SAFE_CALL(CHECK_SYCL_ERROR(h_A = (float *)malloc(size)));
  MY_SAFE_CALL(cuMemHostAlloc((void **)&h_A, size, CU_MEMHOSTALLOC_PORTABLE));
}
