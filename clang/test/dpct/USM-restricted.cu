// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --usm-level=restricted -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/USM-restricted.dp.cpp %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_SAFE_CALL( call) do {\
  int err = call;                \
} while (0)

__constant__ float constData[1234567 * 4];

void foo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  int errorCode;

  cudaStream_t stream;

  /// malloc

  // CHECK: *((void **)&d_A) = cl::sycl::malloc_device(size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  cudaMalloc((void **)&d_A, size);
  // CHECK: errorCode = (*((void **)&d_A) = cl::sycl::malloc_device(size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context()), 0);
  errorCode = cudaMalloc((void **)&d_A, size);
  // CHECK: CUDA_SAFE_CALL((*((void **)&d_A) = cl::sycl::malloc_device(size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context()), 0));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A, size));

  // CHECK: *((void **)&d_A) = cl::sycl::malloc_device(sizeof(cl::sycl::double2) + size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  // CHECK-NEXT: *((void **)&d_A) = cl::sycl::malloc_device(sizeof(cl::sycl::uchar4) + size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  // CHECK-NEXT: *((void **)&d_A) = cl::sycl::malloc_device(sizeof(d_A[0]), dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  cudaMalloc((void **)&d_A, sizeof(double2) + size);
  cudaMalloc((void **)&d_A, sizeof(uchar4) + size);
  cudaMalloc((void **)&d_A, sizeof(d_A[0]));

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context());
  cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
  // CHECK: errorCode = (*((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context()), 0);
  errorCode = cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
  // CHECK: CUDA_SAFE_CALL((*((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context()), 0));
  CUDA_SAFE_CALL(cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault));

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(sizeof(cl::sycl::double2) - size, dpct::get_default_queue().get_context());
  // CHECK-NEXT: *((void **)&h_A) = cl::sycl::malloc_host(sizeof(cl::sycl::uchar4) - size, dpct::get_default_queue().get_context());
  cudaHostAlloc((void **)&h_A, sizeof(double2) - size, cudaHostAllocDefault);
  cudaHostAlloc((void **)&h_A, sizeof(uchar4) - size, cudaHostAllocDefault);

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context());
  cudaMallocHost((void **)&h_A, size);
  // CHECK: errorCode = (*((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context()), 0);
  errorCode = cudaMallocHost((void **)&h_A, size);
  // CHECK: CUDA_SAFE_CALL((*((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context()), 0));
  CUDA_SAFE_CALL(cudaMallocHost((void **)&h_A, size));

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(sizeof(cl::sycl::double2) * size, dpct::get_default_queue().get_context());
  // CHECK-NEXT: *((void **)&h_A) = cl::sycl::malloc_host(sizeof(cl::sycl::uchar4) * size, dpct::get_default_queue().get_context());
  cudaMallocHost((void **)&h_A, sizeof(double2) * size);
  cudaMallocHost((void **)&h_A, sizeof(uchar4) * size);

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context());
  cudaMallocHost(&h_A, size);
  // CHECK: errorCode = (*((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context()), 0);
  errorCode = cudaMallocHost(&h_A, size);
  // CHECK: CUDA_SAFE_CALL((*((void **)&h_A) = cl::sycl::malloc_host(size, dpct::get_default_queue().get_context()), 0));
  CUDA_SAFE_CALL(cudaMallocHost(&h_A, size));

  // CHECK: *((void **)&h_A) = cl::sycl::malloc_host(sizeof(cl::sycl::double2) / size, dpct::get_default_queue().get_context());
  // CHECK-NEXT: *((void **)&h_A) = cl::sycl::malloc_host(sizeof(cl::sycl::uchar4) / size, dpct::get_default_queue().get_context());
  cudaMallocHost(&h_A, sizeof(double2) / size);
  cudaMallocHost(&h_A, sizeof(uchar4) / size);

  // CHECK: *((void **)&d_A) = cl::sycl::malloc_shared(size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  cudaMallocManaged((void **)&d_A, size);
  // CHECK: errorCode = (*((void **)&d_A) = cl::sycl::malloc_shared(size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context()), 0);
  errorCode = cudaMallocManaged((void **)&d_A, size);
  // CHECK: CUDA_SAFE_CALL((*((void **)&d_A) = cl::sycl::malloc_shared(size, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context()), 0));
  CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_A, size));

  // CHECK: *((void **)&d_A) = cl::sycl::malloc_shared(sizeof(cl::sycl::double2) + size + sizeof(cl::sycl::uchar4), dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  // CHECK-NEXT: *((void **)&d_A) = cl::sycl::malloc_shared(sizeof(cl::sycl::double2) * size * sizeof(cl::sycl::uchar4), dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context());
  cudaMallocManaged((void **)&d_A, sizeof(double2) + size + sizeof(uchar4));
  cudaMallocManaged((void **)&d_A, sizeof(double2) * size * sizeof(uchar4));

  /// memcpy

  // CHECK: dpct::get_default_queue_wait().memcpy((void*)(d_A), (void*)(h_A), size).wait();
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: errorCode  = (dpct::get_default_queue_wait().memcpy((void*)(d_A), (void*)(h_A), size).wait(), 0);
  errorCode  = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void*)(d_A), (void*)(h_A), size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

#define SIZE 100
  // CHECK: dpct::get_default_queue_wait().memcpy( (void*)(d_A), (void*)(h_A), sizeof(double)*SIZE*SIZE ).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );

  /// memcpy async

  // CHECK: dpct::get_default_queue_wait().memcpy((void*)(d_A), (void*)(h_A), size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void*)(d_A), (void*)(h_A), size), 0);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void*)(d_A), (void*)(h_A), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice));

  // CHECK: dpct::get_default_queue_wait().memcpy((void*)(d_A), (void*)(h_A), size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void*)(d_A), (void*)(h_A), size), 0);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void*)(d_A), (void*)(h_A), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0));

  // CHECK: stream->memcpy((void*)(d_A), (void*)(h_A), size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
  // CHECK: errorCode = (stream->memcpy((void*)(d_A), (void*)(h_A), size), 0);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memcpy((void*)(d_A), (void*)(h_A), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));

  /// memcpy from symbol

  // CHECK: dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size).wait(), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1));

  // CHECK: dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size).wait(), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  /// memcpy from symbol async

  // CHECK: dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 1), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  // CHECK: dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 2), size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 2), size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 2), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0));

  // CHECK: stream->memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 3), size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: errorCode = (stream->memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 3), size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memcpy((void*)(h_A), (void *)((char *)(constData.get_ptr()) + 3), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream));

  /// memcpy to symbol

  // CHECK: dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size).wait();
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size).wait(), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1));

  // CHECK: dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size).wait();
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size).wait(), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  /// memcpy to symbol async

  // CHECK: dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 1), (void*)(h_A), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  // CHECK: dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 2), (void*)(h_A), size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 2), (void*)(h_A), size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memcpy((void *)((char *)(constData.get_ptr()) + 2), (void*)(h_A), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0));

  // CHECK: stream->memcpy((void *)((char *)(constData.get_ptr()) + 3), (void*)(h_A), size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: errorCode = (stream->memcpy((void *)((char *)(constData.get_ptr()) + 3), (void*)(h_A), size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memcpy((void *)((char *)(constData.get_ptr()) + 3), (void*)(h_A), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream));

  /// memset

  // CHECK: dpct::get_default_queue_wait().memset((void*)(d_A), 23, size).wait();
  cudaMemset(d_A, 23, size);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memset((void*)(d_A), 23, size).wait(), 0);
  errorCode = cudaMemset(d_A, 23, size);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memset((void*)(d_A), 23, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemset(d_A, 23, size));

  /// memset async

  // CHECK: dpct::get_default_queue_wait().memset((void*)(d_A), 23, size);
  cudaMemsetAsync(d_A, 23, size);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memset((void*)(d_A), 23, size), 0);
  errorCode = cudaMemsetAsync(d_A, 23, size);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memset((void*)(d_A), 23, size), 0));
  CUDA_SAFE_CALL(cudaMemsetAsync(d_A, 23, size));

  // CHECK: dpct::get_default_queue_wait().memset((void*)(d_A), 23, size);
  cudaMemsetAsync(d_A, 23, size, 0);
  // CHECK: errorCode = (dpct::get_default_queue_wait().memset((void*)(d_A), 23, size), 0);
  errorCode = cudaMemsetAsync(d_A, 23, size, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue_wait().memset((void*)(d_A), 23, size), 0));
  CUDA_SAFE_CALL(cudaMemsetAsync(d_A, 23, size, 0));

  // CHECK: stream->memset((void*)(d_A), 23, size);
  cudaMemsetAsync(d_A, 23, size, stream);
  // CHECK: errorCode = (stream->memset((void*)(d_A), 23, size), 0);
  errorCode = cudaMemsetAsync(d_A, 23, size, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memset((void*)(d_A), 23, size), 0));
  CUDA_SAFE_CALL(cudaMemsetAsync(d_A, 23, size, stream));

  // CHECK: cl::sycl::free(h_A, dpct::get_default_queue().get_context());
  cudaFreeHost(h_A);
  // CHECK: errorCode = (cl::sycl::free(h_A, dpct::get_default_queue().get_context()), 0);
  errorCode = cudaFreeHost(h_A);
  // CHECK: CUDA_SAFE_CALL((cl::sycl::free(h_A, dpct::get_default_queue().get_context()), 0));
  CUDA_SAFE_CALL(cudaFreeHost(h_A));

  // CHECK: *(&d_A) = h_A;
  cudaHostGetDevicePointer(&d_A, h_A, 0);
  // CHECK: errorCode = (*(&d_A) = h_A, 0);
  errorCode = cudaHostGetDevicePointer(&d_A, h_A, 0);
  // CHECK: CUDA_SAFE_CALL((*(&d_A) = h_A, 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(&d_A, h_A, 0));

  cudaHostRegister(h_A, size, 0);
  // CHECK: errorCode = (0, 0);
  errorCode = cudaHostRegister(h_A, size, 0);
  // CHECK: CUDA_SAFE_CALL((0, 0));
  CUDA_SAFE_CALL(cudaHostRegister(h_A, size, 0));

  cudaHostUnregister(h_A);
  // CHECK: errorCode = (0, 0);
  errorCode = cudaHostUnregister(h_A);
  // CHECK: CUDA_SAFE_CALL((0, 0));
  CUDA_SAFE_CALL(cudaHostUnregister(h_A));
}
