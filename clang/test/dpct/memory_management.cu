// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -output-file=memory_management_outputfile.txt -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memory_management.dp.cpp %s

#include <cuda_runtime.h>

__constant__ float constData[1234567 * 4];

void fooo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  cudaPitchedPtr p_A;
  cudaExtent e;
  float *d_A = NULL;
  cudaStream_t stream;
  cudaMemcpy3DParms parms;
  // CHECK: dpct::dpct_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);
  // CHECK: dpct::dpct_malloc((void **)&d_A, &size, size, size);
  cudaMallocPitch((void **)&d_A, &size, size, size);
  // CHECK: dpct::dpct_malloc(&p_A, e);
  cudaMalloc3D(&p_A, e);
  // CHECK: dpct::dpct_memset(d_A, 0xf, size);
  cudaMemset(d_A, 0xf, size);
  // CHECK: dpct::dpct_memset(d_A, size, 0xf, size, size);
  cudaMemset2D(d_A, size, 0xf, size, size);
  // CHECK: dpct::dpct_memset(p_A, 0xf, e);
  cudaMemset3D(p_A, 0xf, e);

  // CHECK: dpct::async_dpct_memset(d_A, 0xf, size);
  cudaMemsetAsync(d_A, 0xf, size);
  // CHECK: dpct::async_dpct_memset(d_A, 0xf, size);
  cudaMemsetAsync(d_A, 0xf, size, 0);
  // CHECK: dpct::async_dpct_memset(d_A, 0xf, size, *stream);
  cudaMemsetAsync(d_A, 0xf, size, stream);

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

  // CHECK: dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::dpct_memcpy(h_A, size, d_A, size, size, size, dpct::device_to_host);
  cudaMemcpy2D(h_A, size, d_A, size, size, size, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  cudaMemcpy3D(&parms);

  struct cudaMemcpy3DParms *parms_pointer;
  // Followed call can't be processed.
  cudaMemcpy3D(parms_pointer);

  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy(d_A, h_A, size, dpct::host_to_device, *stream);
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);

  // CHECK: dpct::async_dpct_memcpy(h_A, d_A, size, dpct::device_to_host);
  cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, d_A, size, dpct::device_to_host);
  cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, 0);
  // CHECK: dpct::async_dpct_memcpy(h_A, d_A, size, dpct::device_to_host, *stream);
  cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, stream);

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

  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 2, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 3, h_A, size, dpct::host_to_device, *stream);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);

  // CHECK: dpct::async_dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 0, cudaMemcpyHostToDevice);
  // dpct::async_dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 0, cudaMemcpyHostToDevice, 0);
  // dpct::async_dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device, *stream);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 0, cudaMemcpyHostToDevice, stream);

  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 2, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 3, size, dpct::device_to_host, *stream);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);

  // CHECK: dpct::async_dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 0, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 0, cudaMemcpyDeviceToHost, 0);
  // dpct::async_dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host, *stream);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 0, cudaMemcpyDeviceToHost, stream);

  // CHECK: dpct::dpct_free(d_A);
  cudaFree(d_A);
  free(h_A);
}

cudaError_t mallocWrapper(void **buffer, size_t size) {
  if (1) {
    // CHECK:/*
    // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:*/
    // CHECK-NEXT:  return (dpct::dpct_malloc(buffer, size), 0);
    return cudaMalloc(buffer, size);
  }
  if (1) {
    struct cudaPitchedPtr pitch;
    struct cudaExtent pitch_size;
    // CHECK:/*
    // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:*/
    // CHECK-NEXT:  return (dpct::dpct_malloc(&pitch, pitch_size), 0);
    return cudaMalloc3D(&pitch, pitch_size);
  }
  if (1) {
    // CHECK:/*
    // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:*/
    // CHECK-NEXT:  return (dpct::dpct_malloc(buffer, &size, size, size), 0);
    return cudaMallocPitch(buffer, &size, size, size);
  }
}

void checkError(cudaError_t err) {
}

void testCommas() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  float *d_B = NULL;
  cudaPitchedPtr p_A;
  cudaExtent e;
  cudaMemcpy3DParms parms;
  // CHECK:  dpct::dpct_malloc((void **)&d_A, size);
  cudaMalloc((void **)&d_A, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  int err = (dpct::dpct_malloc((void **)&d_A, size), 0);
  cudaError_t err = cudaMalloc((void **)&d_A, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_malloc((void **)&d_A, size), 0));
  checkError(cudaMalloc((void **)&d_A, size));

  // CHECK: dpct::dpct_malloc((void **)&d_A, &size, size, size);
  cudaMallocPitch((void **)&d_A, &size, size, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_malloc((void **)&d_A, &size, size, size), 0);
  err = cudaMallocPitch((void **)&d_A, &size, size, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_malloc((void **)&d_A, &size, size, size), 0));
  checkError(cudaMallocPitch((void **)&d_A, &size, size, size));

  // CHECK: dpct::dpct_malloc(&p_A, e);
  cudaMalloc3D(&p_A, e);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_malloc(&p_A, e), 0);
  err = cudaMalloc3D(&p_A, e);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_malloc(&p_A, e), 0));
  checkError(cudaMalloc3D(&p_A, e));

  // CHECK:  dpct::dpct_memset(d_A, 0xf, size);
  cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memset(d_A, 0xf, size), 0);
  err = cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memset(d_A, 0xf, size), 0));
  checkError(cudaMemset(d_A, 0xf, size));

  // CHECK:  dpct::dpct_memset(d_A, size, 0xf, size, size);
  cudaMemset2D(d_A, size, 0xf, size, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memset(d_A, size, 0xf, size, size), 0);
  err = cudaMemset2D(d_A, size, 0xf, size, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memset(d_A, size, 0xf, size, size), 0));
  checkError(cudaMemset2D(d_A, size, 0xf, size, size));

  // CHECK:  dpct::dpct_memset(p_A, 0xf, e);
  cudaMemset3D(p_A, 0xf, e);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memset(p_A, 0xf, e), 0);
  err = cudaMemset3D(p_A, 0xf, e);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memset(p_A, 0xf, e), 0));
  checkError(cudaMemset3D(p_A, 0xf, e));

  ///////// Host to host
  // CHECK:  dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_host);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_host), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_host), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_device), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_device), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_host), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_host), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

  ///////// Device to Device
  // CHECK:  dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_device);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_device), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_device), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy(h_A, d_A, size, dpct::automatic);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(h_A, d_A, size, dpct::automatic), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(h_A, d_A, size, dpct::automatic), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_host);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_host), 0);
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_host), 0));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToHost));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device), 0);
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device), 0));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_device);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_device), 0);
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_device), 0));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToDevice));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_host);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_host), 0);
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_host), 0));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToHost));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::automatic);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::automatic), 0);
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::automatic), 0));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDefault));

  // CHECK:  dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  cudaMemcpy3D(&parms);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0);
  err = cudaMemcpy3D(&parms);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0));
  checkError(cudaMemcpy3D(&parms));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device), 0);
  err = cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device), 0));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyHostToDevice));

  // CHECK:  dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device), 0);
  err = cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device), 0));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::device_to_device);
  cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::device_to_device), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(constData.get_ptr(), h_A, size, dpct::device_to_device), 0));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyDeviceToDevice));

  // CHECK:  dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::device_to_device);
  cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::device_to_device), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::device_to_device), 0));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::automatic);
  cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::automatic), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::automatic), 0));
  checkError(cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDefault));

  // CHECK:  dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::automatic);
  cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::automatic), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::automatic), 0));
  checkError(cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDefault));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), d_B, size);
  cudaMemcpyToSymbol(constData, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy(constData.get_ptr(), d_B, size), 0);
  err = cudaMemcpyToSymbol(constData, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(constData.get_ptr(), d_B, size), 0));
  checkError(cudaMemcpyToSymbol(constData, d_B, size));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, constData, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host), 0);
  err = cudaMemcpyFromSymbol(h_A, constData, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host), 0));
  checkError(cudaMemcpyFromSymbol(h_A, constData, size, 0, cudaMemcpyDeviceToHost));

  // CHECK:  dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host), 0);
  err = cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host), 0));
  checkError(cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(d_B, constData.get_ptr(), size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_B, constData, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_B, constData.get_ptr(), size, dpct::device_to_device), 0);
  err = cudaMemcpyFromSymbol(d_B, constData, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(d_B, constData.get_ptr(), size, dpct::device_to_device), 0));
  checkError(cudaMemcpyFromSymbol(d_B, constData, size, 0, cudaMemcpyDeviceToDevice));


  // CHECK:  dpct::dpct_memcpy(d_B, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_B, constData, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_B, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_device), 0);
  err = cudaMemcpyFromSymbol(d_B, constData, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(d_B, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_device), 0));
  checkError(cudaMemcpyFromSymbol(d_B, constData, size, 1, cudaMemcpyDeviceToDevice));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy(h_A, constData.get_ptr(), size), 0);
  err = cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(h_A, constData.get_ptr(), size), 0));
  checkError(cudaMemcpyFromSymbol(h_A, constData, size));

  // CHECK: dpct::dpct_free(d_A);
  cudaFree(d_A);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_free(d_A), 0);
  err = cudaFree(d_A);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_free(d_A), 0));
  checkError(cudaFree(d_A));
  // CHECK:  free(h_A);
  free(h_A);
}

// CHECK:  dpct::device_memory<float, 1> d_A(1234567);
// CHECK:  dpct::device_memory<float, 1> d_B(1234567);
static __device__ float d_A[1234567];
static __device__ float d_B[1234567];

void testCommas_in_device_memory() {
  size_t size = 1234567 * sizeof(float);
  cudaError_t err;
  float *h_A = (float *)malloc(size);

  // CHECK:  dpct::dpct_memset(d_A.get_ptr(), 0xf, size);
  cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memset(d_A.get_ptr(), 0xf, size), 0);
  err = cudaMemset(d_A, 0xf, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memset(d_A.get_ptr(), 0xf, size), 0));
  checkError(cudaMemset(d_A, 0xf, size));

  ///////// Host to host
  // CHECK:  dpct::dpct_memcpy(h_A, h_A, size, dpct::host_to_host);
  cudaMemcpy(h_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(h_A, h_A, size, dpct::host_to_host), 0);
  err = cudaMemcpy(h_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(h_A, h_A, size, dpct::host_to_host), 0));
  checkError(cudaMemcpy(h_A, h_A, size, cudaMemcpyHostToHost));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device), 0);
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device), 0));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

  ///////// Device to Device
  // CHECK:  dpct::dpct_memcpy(d_B.get_ptr(), d_A.get_ptr(), size, dpct::device_to_device);
  cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_B.get_ptr(), d_A.get_ptr(), size, dpct::device_to_device), 0);
  err = cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_B.get_ptr(), d_A.get_ptr(), size, dpct::device_to_device), 0));
  checkError(cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::automatic);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::automatic), 0);
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::automatic), 0));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device), 0);
  err = cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device), 0));
  checkError(cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device);
  cudaMemcpyToSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  err = cudaMemcpyToSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device), 0));
  checkError(cudaMemcpyToSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy(h_A, d_B.get_ptr(), size, dpct::automatic);
  cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy(h_A, d_B.get_ptr(), size, dpct::automatic), 0);
  err = cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(h_A, d_B.get_ptr(), size, dpct::automatic), 0));
  checkError(cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy(h_A, d_B.get_ptr(), size);
  cudaMemcpyToSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy(h_A, d_B.get_ptr(), size), 0);
  err = cudaMemcpyToSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(h_A, d_B.get_ptr(), size), 0));
  checkError(cudaMemcpyToSymbol(h_A, d_B, size));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, d_A, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host), 0);
  err = cudaMemcpyFromSymbol(h_A, d_A, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host), 0));
  checkError(cudaMemcpyFromSymbol(h_A, d_A, size, 0, cudaMemcpyDeviceToHost));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  err = (dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device), 0);
  err = cudaMemcpyFromSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device), 0));
  checkError(cudaMemcpyFromSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy(h_A, d_B.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   err = (dpct::dpct_memcpy(h_A, d_B.get_ptr(), size), 0);
  err = cudaMemcpyFromSymbol(h_A, d_B, size);
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   checkError((dpct::dpct_memcpy(h_A, d_B.get_ptr(), size), 0));
  checkError(cudaMemcpyFromSymbol(h_A, d_B, size));

  void *p_addr;
  // CHECK:  *(&p_addr) = d_A.get_ptr();
  cudaGetSymbolAddress(&p_addr, d_A);

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  err = (*(&p_addr) = d_A.get_ptr(), 0);
  err = cudaGetSymbolAddress(&p_addr, d_A);

  // CHECK: /*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:checkError((*(&p_addr) = d_A.get_ptr(), 0));
  checkError(cudaGetSymbolAddress(&p_addr, d_A));

  size_t size2;
  // CHECK: size2 = d_A.get_size();
  cudaGetSymbolSize(&size2, d_A);

  // CHECK: /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  err = (size2 = d_A.get_size(), 0);
  err = cudaGetSymbolSize(&size2, d_A);

  // CHECK: /*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:checkError((size2 = d_A.get_size(), 0));
  checkError(cudaGetSymbolSize(&size2, d_A));

  int* a;
  cudaStream_t stream;
  int deviceID = 0;
  // CHECK:/*
  // CHECK-NEXT:DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT:*/
  cudaMemPrefetchAsync (a, 100, deviceID, stream);

  // CHECK:  free(h_A);
  free(h_A);
}

#define CUDA_CHECK(call)                                                           \
    if ((call) != cudaSuccess) { \
        exit(-1); \
    }

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {}


template<typename T>
void uninstantiated_template_call(const T * d_data, size_t width, size_t height) {
  size_t datasize = width * height;
  T * data = new T[datasize];
  cudaMemcpy3DParms parms;
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  assert_cuda((dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host), 0));
  assert_cuda(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK: dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host);
  cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

#define DATAMACRO data+32*32

  // CHECK: dpct::dpct_memcpy(DATAMACRO, d_data, datasize * sizeof(T), dpct::device_to_host);
  cudaMemcpy(DATAMACRO, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(32*32+DATAMACRO, d_data, datasize * sizeof(T), dpct::device_to_host);
  cudaMemcpy(32*32+DATAMACRO, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  checkError((dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host), 0));
  checkError(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: int err = (dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host), 0);
  cudaError_t err = cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: CUDA_CHECK((dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host), 0));
  CUDA_CHECK(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: checkCudaErrors((dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host), 0));
  checkCudaErrors(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK: #define CUDAMEMCPY dpct::dpct_memcpy
  // CHECK-NEXT: CUDAMEMCPY(data, d_data, datasize * sizeof(T), dpct::device_to_host);
  #define CUDAMEMCPY cudaMemcpy
  CUDAMEMCPY(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:  assert_cuda((dpct::dpct_memcpy(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host), 0));
  assert_cuda(cudaMemcpy2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost));

  // CHECK: dpct::dpct_memcpy(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host);
  cudaMemcpy2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(DATAMACRO, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host);
  cudaMemcpy2D(DATAMACRO, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(32*32+DATAMACRO, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host);
  cudaMemcpy2D(32*32+DATAMACRO, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: CUDA_CHECK((dpct::dpct_memcpy(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host), 0));
  CUDA_CHECK(cudaMemcpy2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost));

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: checkCudaErrors((dpct::dpct_memcpy(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host), 0));
  checkCudaErrors(cudaMemcpy2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost));

  // CHECK: #define CUDAMEMCPY2D dpct::dpct_memcpy
  // CHECK-NEXT: CUDAMEMCPY2D(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host);
  #define CUDAMEMCPY2D cudaMemcpy2D
  CUDAMEMCPY2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: CUDA_CHECK((dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0));
  CUDA_CHECK(cudaMemcpy3D(&parms));

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT: checkCudaErrors((dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1), 0));
  checkCudaErrors(cudaMemcpy3D(&parms));

  delete[] data;
}

#define N 1024
void test_segmentation_fault() {
  float *buffer;
  /*
  * Original code in getSizeString():
  * "SizeExpr->getBeginLoc()" cannot get the real SourceLocation of "N*sizeof(float)",
  * and results in boundary violation in "dpctGlobalInfo::getSourceManager().getCharacterData(SizeBegin)"
  * and fails with segmentation fault.
  * https://jira.devtools.intel.com/browse/CTST-527
  * https://jira.devtools.intel.com/browse/CTST-528
  */
  cudaMalloc(&buffer, N*sizeof(float));
}

// CHECK: dpct::device_memory<uint32_t, 1> d_error(1);
static __device__ uint32_t d_error[1];

void test_foo(){
  // CHECK: dpct::dpct_memset(d_error.get_ptr(), 0, sizeof(uint32_t));
  cudaMemset(d_error, 0, sizeof(uint32_t));
}

void foobar() {
  int errorCode;

  cudaChannelFormatDesc desc;
  cudaExtent extent;
  unsigned int flags;
  cudaArray_t array;

  // CHECK: desc = array->get_channel();
  // CHECK: extent = array->get_range();
  // CHECK: flags = 0;
  cudaArrayGetInfo(&desc, &extent, &flags, array);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError(([&](){
  // CHECK-NEXT:   desc = array->get_channel();
  // CHECK-NEXT:   extent = array->get_range();
  // CHECK-NEXT:   flags = 0;
  // CHECK-NEXT:   }(), 0));
  checkError(cudaArrayGetInfo(&desc, &extent, &flags, array));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = ([&](){
  // CHECK-NEXT:   desc = array->get_channel();
  // CHECK-NEXT:   extent = array->get_range();
  // CHECK-NEXT:   flags = 0;
  // CHECK-NEXT:   }(), 0);
  errorCode = cudaArrayGetInfo(&desc, &extent, &flags, array);

  int host;
  // CHECK: flags = 0;
  cudaHostGetFlags(&flags, &host);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError((flags = 0, 0));
  checkError(cudaHostGetFlags(&flags, &host));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = (flags = 0, 0);
  errorCode = cudaHostGetFlags(&flags, &host);

  int *devPtr;
  size_t count;
  // CHECK: pi_mem_advice advice;
  cudaMemoryAdvise advice;
  int device;
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaMemAdvise(devPtr, count, advice, device);
  cudaMemAdvise(devPtr, count, advice, device);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError(cudaMemAdvise(devPtr, count, advice, device));
  checkError(cudaMemAdvise(devPtr, count, advice, device));

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = cudaMemAdvise(devPtr, count, advice, device);
  errorCode = cudaMemAdvise(devPtr, count, advice, device);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device);
  cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError(cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device));
  checkError(cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device));

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device);
  errorCode = cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device);
}

// CHECK: void copy_dir_1 (dpct::memcpy_direction kind) {}
// CHECK-NEXT: void copy_dir_2 (dpct::memcpy_direction kind) {}
// CHECK-NEXT: void copy_dir_3 (dpct::memcpy_direction kind) {}
void copy_dir_1 (cudaMemcpyKind kind) {}
void copy_dir_2 (enum cudaMemcpyKind kind) {}
void copy_dir_3 (enum    cudaMemcpyKind kind) {}

// CHECK: void copy_dir_1 (int kind) {}
// CHECK-NEXT: void copy_dir_2 (int kind) {}
// CHECK-NEXT: void copy_dir_3 (int kind) {}
void copy_dir_1 (cudaComputeMode kind) {}
void copy_dir_2 (enum cudaComputeMode kind) {}
void copy_dir_3 (enum    cudaComputeMode kind) {}
