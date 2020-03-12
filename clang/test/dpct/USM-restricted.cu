// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=restricted -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/USM-restricted.dp.cpp %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_SAFE_CALL( call) do {\
  int err = call;                \
} while (0)

__constant__ float constData[1234567 * 4];

int foo_b(int a){
  return 0;
}

void foo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  int errorCode;

  cudaPitchedPtr p_A;
  cudaExtent e;
  cudaMemcpy3DParms parms;
  cudaStream_t stream;

  /// malloc
  // CHECK: d_A = (float *)sycl::malloc_device(size, dpct::get_current_device(), dpct::get_default_context());
  cudaMalloc((void **)&d_A, size);
  // CHECK: errorCode = (d_A = (float *)sycl::malloc_device(size, dpct::get_current_device(), dpct::get_default_context()), 0);
  errorCode = cudaMalloc((void **)&d_A, size);
  // CHECK: CUDA_SAFE_CALL((d_A = (float *)sycl::malloc_device(size, dpct::get_current_device(), dpct::get_default_context()), 0));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A, size));

  // CHECK: d_A = (float *)sycl::malloc_device(sizeof(sycl::double2) + size, dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: d_A = (float *)sycl::malloc_device(sizeof(sycl::uchar4) + size, dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: d_A = (float *)sycl::malloc_device(sizeof(d_A[0]), dpct::get_current_device(), dpct::get_default_context());
  cudaMalloc((void **)&d_A, sizeof(double2) + size);
  cudaMalloc((void **)&d_A, sizeof(uchar4) + size);
  cudaMalloc((void **)&d_A, sizeof(d_A[0]));
  
  // CHECK: dpct::dpct_malloc((void **)&d_A, &size, size, size);
  cudaMallocPitch((void **)&d_A, &size, size, size);
  // CHECK: dpct::dpct_malloc(&p_A, e);
  cudaMalloc3D(&p_A, e);

  // CHECK: h_A = (float *)sycl::malloc_host(size, dpct::get_default_context());
  cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
  // CHECK: errorCode = (h_A = (float *)sycl::malloc_host(size, dpct::get_default_context()), 0);
  errorCode = cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault);
  // CHECK: CUDA_SAFE_CALL((h_A = (float *)sycl::malloc_host(size, dpct::get_default_context()), 0));
  CUDA_SAFE_CALL(cudaHostAlloc((void **)&h_A, size, cudaHostAllocDefault));

  // CHECK: h_A = (float *)sycl::malloc_host(sizeof(sycl::double2) - size, dpct::get_default_context());
  // CHECK-NEXT: h_A = (float *)sycl::malloc_host(sizeof(sycl::uchar4) - size, dpct::get_default_context());
  cudaHostAlloc((void **)&h_A, sizeof(double2) - size, cudaHostAllocDefault);
  cudaHostAlloc((void **)&h_A, sizeof(uchar4) - size, cudaHostAllocDefault);

  // CHECK: h_A = (float *)sycl::malloc_host(size, dpct::get_default_context());
  cudaMallocHost((void **)&h_A, size);
  // CHECK: errorCode = (h_A = (float *)sycl::malloc_host(size, dpct::get_default_context()), 0);
  errorCode = cudaMallocHost((void **)&h_A, size);
  // CHECK: CUDA_SAFE_CALL((h_A = (float *)sycl::malloc_host(size, dpct::get_default_context()), 0));
  CUDA_SAFE_CALL(cudaMallocHost((void **)&h_A, size));

  // CHECK: h_A = (float *)sycl::malloc_host(sizeof(sycl::double2) * size, dpct::get_default_context());
  // CHECK-NEXT: h_A = (float *)sycl::malloc_host(sizeof(sycl::uchar4) * size, dpct::get_default_context());
  cudaMallocHost((void **)&h_A, sizeof(double2) * size);
  cudaMallocHost((void **)&h_A, sizeof(uchar4) * size);

  // CHECK: h_A = (float *)sycl::malloc_host(size, dpct::get_default_context());
  cudaMallocHost(&h_A, size);
  // CHECK: errorCode = (h_A = (float *)sycl::malloc_host(size, dpct::get_default_context()), 0);
  errorCode = cudaMallocHost(&h_A, size);
  // CHECK: CUDA_SAFE_CALL((h_A = (float *)sycl::malloc_host(size, dpct::get_default_context()), 0));
  CUDA_SAFE_CALL(cudaMallocHost(&h_A, size));

  // CHECK: h_A = (float *)sycl::malloc_host(sizeof(sycl::double2) / size, dpct::get_default_context());
  // CHECK-NEXT: h_A = (float *)sycl::malloc_host(sizeof(sycl::uchar4) / size, dpct::get_default_context());
  cudaMallocHost(&h_A, sizeof(double2) / size);
  cudaMallocHost(&h_A, sizeof(uchar4) / size);

  // CHECK: d_A = (float *)sycl::malloc_shared(size, dpct::get_current_device(), dpct::get_default_context());
  cudaMallocManaged((void **)&d_A, size);
  // CHECK: errorCode = (d_A = (float *)sycl::malloc_shared(size, dpct::get_current_device(), dpct::get_default_context()), 0);
  errorCode = cudaMallocManaged((void **)&d_A, size);
  // CHECK: CUDA_SAFE_CALL((d_A = (float *)sycl::malloc_shared(size, dpct::get_current_device(), dpct::get_default_context()), 0));
  CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_A, size));

  // CHECK: d_A = (float *)sycl::malloc_shared(sizeof(sycl::double2) + size + sizeof(sycl::uchar4), dpct::get_current_device(), dpct::get_default_context());
  // CHECK-NEXT: d_A = (float *)sycl::malloc_shared(sizeof(sycl::double2) * size * sizeof(sycl::uchar4), dpct::get_current_device(), dpct::get_default_context());
  cudaMallocManaged((void **)&d_A, sizeof(double2) + size + sizeof(uchar4));
  cudaMallocManaged((void **)&d_A, sizeof(double2) * size * sizeof(uchar4));

  /// memcpy

  // CHECK: sycl::queue& q_ct0 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct0.wait();
  // CHECK-NEXT: q_ct0.memcpy(d_A, h_A, size).wait();
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: errorCode  = (q_ct0.memcpy(d_A, h_A, size).wait(), 0);
  errorCode  = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(d_A, h_A, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
#define MACRO_A(x) size
#define MACRO_A2(x) MACRO_A(x)
#define MACRO_B size
#define MACOR_C(x) cudaMemcpyDeviceToHost
#define CUDA_SAFE_CALL2(x) CUDA_SAFE_CALL(x)
  //CHECK: /*
  //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: CUDA_SAFE_CALL2((dpct::get_default_queue().memcpy(d_A, h_A, size).wait(), 0));
  CUDA_SAFE_CALL2(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  //CHECK: /*
  //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: CUDA_SAFE_CALL2((dpct::get_default_queue().memcpy(d_A, h_A, MACRO_B).wait(), 0));
  CUDA_SAFE_CALL2(cudaMemcpy(d_A, h_A, MACRO_B, cudaMemcpyDeviceToHost));
  //CHECK: /*
  //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: CUDA_SAFE_CALL2((dpct::get_default_queue().memcpy(d_A, h_A, MACRO_A2(1)).wait(), 0));
  CUDA_SAFE_CALL2(cudaMemcpy(d_A, h_A, MACRO_A2(1), MACOR_C(1)));
  //CHECK: /*
  //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT: */
  //CHECK-NEXT: CUDA_SAFE_CALL2((dpct::get_default_queue().memcpy(d_A, h_A, foo_b(1)).wait(), 0));
  CUDA_SAFE_CALL2(cudaMemcpy(d_A, h_A, foo_b(1), MACOR_C(1)));

#define SIZE 100
  // CHECK: dpct::get_default_queue().memcpy( d_A, h_A, sizeof(double)*SIZE*SIZE ).wait();
  cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );

  /// memcpy async

  // CHECK: dpct::get_default_queue().memcpy(d_A, h_A, size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy(d_A, h_A, size), 0);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(d_A, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice));

  // CHECK: dpct::get_default_queue().memcpy(d_A, h_A, size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy(d_A, h_A, size), 0);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(d_A, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0));

  // CHECK: stream->memcpy(d_A, h_A, size);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
  // CHECK: errorCode = (stream->memcpy(d_A, h_A, size), 0);
  errorCode = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memcpy(d_A, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));

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

  // CHECK: sycl::queue& q_ct1 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct1.wait();
  // CHECK-NEXT: q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: errorCode = (q_ct1.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait(), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1));

  // CHECK: sycl::queue& q_ct2 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct2.wait();
  // CHECK-NEXT: q_ct2.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (q_ct2.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait(), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  /// memcpy from symbol async

  // CHECK: dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  // CHECK: dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 2, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 2, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 2, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0));

  // CHECK: stream->memcpy(h_A, (char *)(constData.get_ptr()) + 3, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: errorCode = (stream->memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream));

  /// memcpy to symbol
  // CHECK: sycl::queue& q_ct3 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct3.wait();
  // CHECK-NEXT: q_ct3.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: errorCode = (q_ct3.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait(), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1));

  // CHECK: sycl::queue& q_ct4 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct4.wait();
  // CHECK-NEXT: q_ct4.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (q_ct4.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait(), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  /// memcpy to symbol async

  // CHECK: dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  // CHECK: dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 2, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 2, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 2, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0));

  // CHECK: stream->memcpy((char *)(constData.get_ptr()) + 3, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: errorCode = (stream->memcpy((char *)(constData.get_ptr()) + 3, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memcpy((char *)(constData.get_ptr()) + 3, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream));

  /// memset

  // CHECK: sycl::queue& q_ct5 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct5.wait();
  // CHECK-NEXT: q_ct5.memset(d_A, 23, size).wait();
  cudaMemset(d_A, 23, size);
  // CHECK: errorCode = (q_ct5.memset(d_A, 23, size).wait(), 0);
  errorCode = cudaMemset(d_A, 23, size);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memset(d_A, 23, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemset(d_A, 23, size));

  /// memset async

  // CHECK: dpct::get_default_queue().memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size);
  // CHECK: errorCode = (dpct::get_default_queue().memset(d_A, 23, size), 0);
  errorCode = cudaMemsetAsync(d_A, 23, size);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memset(d_A, 23, size), 0));
  CUDA_SAFE_CALL(cudaMemsetAsync(d_A, 23, size));

  // CHECK: dpct::get_default_queue().memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size, 0);
  // CHECK: errorCode = (dpct::get_default_queue().memset(d_A, 23, size), 0);
  errorCode = cudaMemsetAsync(d_A, 23, size, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memset(d_A, 23, size), 0));
  CUDA_SAFE_CALL(cudaMemsetAsync(d_A, 23, size, 0));

  // CHECK: stream->memset(d_A, 23, size);
  cudaMemsetAsync(d_A, 23, size, stream);
  // CHECK: errorCode = (stream->memset(d_A, 23, size), 0);
  errorCode = cudaMemsetAsync(d_A, 23, size, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memset(d_A, 23, size), 0));
  CUDA_SAFE_CALL(cudaMemsetAsync(d_A, 23, size, stream));
  
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

  // CHECK: sycl::free(h_A, dpct::get_default_context());
  cudaFreeHost(h_A);
  // CHECK: errorCode = (sycl::free(h_A, dpct::get_default_context()), 0);
  errorCode = cudaFreeHost(h_A);
  // CHECK: CUDA_SAFE_CALL((sycl::free(h_A, dpct::get_default_context()), 0));
  CUDA_SAFE_CALL(cudaFreeHost(h_A));

  // CHECK: *(&d_A) = h_A;
  cudaHostGetDevicePointer(&d_A, h_A, 0);
  // CHECK: errorCode = (*(&d_A) = h_A, 0);
  errorCode = cudaHostGetDevicePointer(&d_A, h_A, 0);
  // CHECK: CUDA_SAFE_CALL((*(&d_A) = h_A, 0));
  CUDA_SAFE_CALL(cudaHostGetDevicePointer(&d_A, h_A, 0));

  cudaHostRegister(h_A, size, 0);
  // CHECK: errorCode = 0;
  errorCode = cudaHostRegister(h_A, size, 0);
  // CHECK: CUDA_SAFE_CALL(0);
  CUDA_SAFE_CALL(cudaHostRegister(h_A, size, 0));

  cudaHostUnregister(h_A);
  // CHECK: errorCode = 0;
  errorCode = cudaHostUnregister(h_A);
  // CHECK: CUDA_SAFE_CALL(0);
  CUDA_SAFE_CALL(cudaHostUnregister(h_A));
}


template <typename T>
int foo2() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  int errorCode;

  cudaStream_t stream;
  /// memcpy from symbol

  // CHECK: sycl::queue& q_ct6 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct6.wait();
  // CHECK-NEXT: q_ct6.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: errorCode = (q_ct6.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait(), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1));

  // CHECK: sycl::queue& q_ct7 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct7.wait();
  // CHECK-NEXT: q_ct7.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (q_ct7.memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait(), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  // CHECK: sycl::queue& q_ct8 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct8.wait();
  // CHECK-NEXT: q_ct8.memcpy(h_A, constData.get_ptr(), size).wait();
  cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK: errorCode = (q_ct8.memcpy(h_A, constData.get_ptr(), size).wait(), 0);
  errorCode = cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, constData.get_ptr(), size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(h_A, constData, size));

  /// memcpy from symbol async

  // CHECK: dpct::get_default_queue().memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy(h_A, constData.get_ptr(), size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, constData.get_ptr(), size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size));
  
  // CHECK: dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 1);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 1));

  // CHECK: dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 1, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  // CHECK: dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 2, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 2, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(h_A, (char *)(constData.get_ptr()) + 2, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0));

  // CHECK: stream->memcpy(h_A, (char *)(constData.get_ptr()) + 3, size);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: errorCode = (stream->memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0);
  errorCode = cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memcpy(h_A, (char *)(constData.get_ptr()) + 3, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream));

  /// memcpy to symbol
  // CHECK: sycl::queue& q_ct9 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct9.wait();
  // CHECK-NEXT: q_ct9.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: errorCode = (q_ct9.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait(), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1));

  // CHECK: sycl::queue& q_ct10 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct10.wait();
  // CHECK-NEXT: q_ct10.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait();
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (q_ct10.memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait(), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  // CHECK: sycl::queue& q_ct11 = dpct::get_default_queue();
  // CHECK-NEXT: q_ct11.wait();
  // CHECK-NEXT: q_ct11.memcpy(constData.get_ptr(), h_A, size).wait();
  cudaMemcpyToSymbol(constData, h_A, size);
  // CHECK: errorCode = (q_ct11.memcpy(constData.get_ptr(), h_A, size).wait(), 0);
  errorCode = cudaMemcpyToSymbol(constData, h_A, size);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(constData.get_ptr(), h_A, size).wait(), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(constData, h_A, size));

  /// memcpy to symbol async
  // CHECK: dpct::get_default_queue().memcpy(constData.get_ptr(), h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy(constData.get_ptr(), h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy(constData.get_ptr(), h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size));

  // CHECK: dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1));

  // CHECK: dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 1, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  // CHECK: dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 2, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: errorCode = (dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 2, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: CUDA_SAFE_CALL((dpct::get_default_queue().memcpy((char *)(constData.get_ptr()) + 2, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0));

  // CHECK: stream->memcpy((char *)(constData.get_ptr()) + 3, h_A, size);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: errorCode = (stream->memcpy((char *)(constData.get_ptr()) + 3, h_A, size), 0);
  errorCode = cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);
  // CHECK: CUDA_SAFE_CALL((stream->memcpy((char *)(constData.get_ptr()) + 3, h_A, size), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream));
}

template int foo2<float>();
template int foo2<int>();
