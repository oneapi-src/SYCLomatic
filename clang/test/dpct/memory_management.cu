// FIXME
// UNSUPPORTED: system-windows
// RUN: dpct --no-dpcpp-extensions=device_info --format-range=none --usm-level=none -out-root %T/memory_management %s --cuda-include-path="%cuda-path/include" -output-file=memory_management_outputfile.txt -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memory_management/memory_management.dp.cpp %s

#include <cuda_runtime.h>
#include<cuda.h>

__constant__ float constData[123 * 4];

// CHECK: template<typename T>
//CHECK-NEXT: void test(){
//CHECK-NEXT:   int i = 0;
//CHECK-NEXT:   T** ptr;
//CHECK-NEXT:   T* array[10];
//CHECK-NEXT:   ptr[i] = (T *)dpct::dpct_malloc(10 * sizeof(T));
//CHECK-NEXT:   ptr[i] = (T *)dpct::dpct_malloc(10 * sizeof(T));
//CHECK-NEXT:   array[i] = (T *)dpct::dpct_malloc(10 * sizeof(T));
//CHECK-NEXT: }
template<typename T>
void test(){
  int i = 0;
  T** ptr;
  T* array[10];
  cudaMalloc(&ptr[i], 10 * sizeof(T));
  cudaMalloc(&(ptr[i]), 10 * sizeof(T));
  cudaMalloc(&array[i], 10 * sizeof(T));
}

void fooo() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  cudaPitchedPtr p_A;
  cudaExtent e;
  float *d_A = NULL;
  cudaStream_t stream;
  cudaMemcpy3DParms parms;
  // CHECK: d_A = (float *)dpct::dpct_malloc(size);
  cudaMalloc((void **)&d_A, size);
  // CHECK: d_A = (float *)dpct::dpct_malloc(size);
  cudaMalloc(reinterpret_cast<void **>(&d_A), size);
  // CHECK: d_A = (float *)dpct::dpct_malloc(size, size, size);
  cudaMallocPitch((void **)&d_A, &size, size, size);
  // CHECK: p_A = dpct::dpct_malloc(e);
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
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr()) + 2, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 2, cudaMemcpyHostToDevice, 0);
  // CHECK: dpct::async_dpct_memcpy((char *)(constData.get_ptr(*stream)) + 3, h_A, size, dpct::host_to_device, *stream);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 3, cudaMemcpyHostToDevice, stream);

  // CHECK: dpct::async_dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK: dpct::async_dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync("constData", h_A, size, 0, cudaMemcpyHostToDevice);
  // dpct::async_dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 0, cudaMemcpyHostToDevice, 0);
  // dpct::async_dpct_memcpy(constData.get_ptr(stream), h_A, size, dpct::host_to_device, *stream);
  cudaMemcpyToSymbolAsync(constData, h_A, size, 0, cudaMemcpyHostToDevice, stream);

  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size, 1, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 2, size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 2, cudaMemcpyDeviceToHost, 0);
  // CHECK: dpct::async_dpct_memcpy(h_A, (char *)(constData.get_ptr(*stream)) + 3, size, dpct::device_to_host, *stream);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 3, cudaMemcpyDeviceToHost, stream);

  // CHECK: dpct::async_dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 0, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, "constData", size, 0, cudaMemcpyDeviceToHost);
  // CHECK: dpct::async_dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 0, cudaMemcpyDeviceToHost, 0);
  // dpct::async_dpct_memcpy(h_A, constData.get_ptr(*stream)), size, dpct::device_to_host, *stream);
  cudaMemcpyFromSymbolAsync(h_A, constData, size, 0, cudaMemcpyDeviceToHost, stream);

  // CHECK: dpct::dpct_free(d_A);
  cudaFree(d_A);
  free(h_A);
}

cudaError_t mallocWrapper(void **buffer, size_t size) {
  if (1) {
    // CHECK:  return CHECK_SYCL_ERROR(*buffer = dpct::dpct_malloc(size));
    return cudaMalloc(buffer, size);
  }
  if (1) {
    struct cudaPitchedPtr pitch;
    struct cudaExtent pitch_size;
    // CHECK:  return CHECK_SYCL_ERROR(pitch = dpct::dpct_malloc(pitch_size));
    return cudaMalloc3D(&pitch, pitch_size);
  }
  if (1) {
    // CHECK:  return CHECK_SYCL_ERROR(*buffer = dpct::dpct_malloc(size, size, size));
    return cudaMallocPitch(buffer, &size, size, size);
  }
}

void checkError(cudaError_t err) {
}

void cuCheckError(CUresult err) {
}
//CHECK: #define PITCH(a,b,c,d) a = (float *)dpct::dpct_malloc(b, c, d);
#define PITCH(a,b,c,d) cudaMallocPitch(a, b, c, d);

void testCommas() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  float *d_B = NULL;
  cudaPitchedPtr p_A;
  cudaExtent e;
  cudaMemcpy3DParms parms;
  // CHECK:  d_A = (float *)dpct::dpct_malloc(size);
  cudaMalloc((void **)&d_A, size);
  // CHECK:  dpct::err0 err = CHECK_SYCL_ERROR(d_A = (float *)dpct::dpct_malloc(size));
  cudaError_t err = cudaMalloc((void **)&d_A, size);
  // CHECK:  checkError(CHECK_SYCL_ERROR(d_A = (float *)dpct::dpct_malloc(size)));
  checkError(cudaMalloc((void **)&d_A, size));

  // CHECK: d_A = (float *)dpct::dpct_malloc(size, size, size);
  cudaMallocPitch((void **)&d_A, &size, size, size);

  // CHECK: PITCH((void **)&d_A, &size, size, size);
  PITCH((void **)&d_A, &size, size, size);
  int sz;
  // CHECK: d_A = (float *)dpct::dpct_malloc(*(size_t *)&size, size, size);
  cudaMallocPitch((void **)&d_A, (size_t *)&size, size, size);
  // CHECK: d_A = (float *)dpct::dpct_malloc(*(size_t *)&sz, size, size);
  cudaMallocPitch((void **)&d_A, (size_t *)&sz, size, size);
  // CHECK:  err = CHECK_SYCL_ERROR(d_A = (float *)dpct::dpct_malloc(size, size, size));
  err = cudaMallocPitch((void **)&d_A, &size, size, size);
  // CHECK:  checkError(CHECK_SYCL_ERROR(d_A = (float *)dpct::dpct_malloc(size, size, size)));
  checkError(cudaMallocPitch((void **)&d_A, &size, size, size));

  // CHECK: p_A = dpct::dpct_malloc(e);
  cudaMalloc3D(&p_A, e);
  // CHECK:  err = CHECK_SYCL_ERROR(p_A = dpct::dpct_malloc(e));
  err = cudaMalloc3D(&p_A, e);
  // CHECK:  checkError(CHECK_SYCL_ERROR(p_A = dpct::dpct_malloc(e)));
  checkError(cudaMalloc3D(&p_A, e));

  // CHECK:  dpct::dpct_memset(d_A, 0xf, size);
  cudaMemset(d_A, 0xf, size);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memset(d_A, 0xf, size));
  err = cudaMemset(d_A, 0xf, size);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memset(d_A, 0xf, size)));
  checkError(cudaMemset(d_A, 0xf, size));

  // CHECK:  dpct::dpct_memset(d_A, size, 0xf, size, size);
  cudaMemset2D(d_A, size, 0xf, size, size);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memset(d_A, size, 0xf, size, size));
  err = cudaMemset2D(d_A, size, 0xf, size, size);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memset(d_A, size, 0xf, size, size)));
  checkError(cudaMemset2D(d_A, size, 0xf, size, size));

  // CHECK:  dpct::dpct_memset(p_A, 0xf, e);
  cudaMemset3D(p_A, 0xf, e);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memset(p_A, 0xf, e));
  err = cudaMemset3D(p_A, 0xf, e);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memset(p_A, 0xf, e)));
  checkError(cudaMemset3D(p_A, 0xf, e));

  ///////// Host to host
  // CHECK:  dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_host);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_host));
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_host)));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToHost));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_device));
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, h_A, size, dpct::host_to_device)));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_host));
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_host)));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

  ///////// Device to Device
  // CHECK:  dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_device);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_device));
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A, size, dpct::device_to_device)));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy(h_A, d_A, size, dpct::automatic);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A, size, dpct::automatic));
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A, size, dpct::automatic)));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_host);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToHost);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_host));
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToHost);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_host)));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToHost));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device));
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::host_to_device)));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyHostToDevice));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_device);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_device));
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToDevice);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_device)));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToDevice));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_host);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToHost);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_host));
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::device_to_host)));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDeviceToHost));

  // CHECK:  dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::automatic);
  cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDefault);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::automatic));
  err = cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDefault);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A, size, h_A, size, size, size, dpct::automatic)));
  checkError(cudaMemcpy2D(d_A, size, h_A, size, size, size, cudaMemcpyDefault));

  // CHECK:  dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  cudaMemcpy3D(&parms);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1));
  err = cudaMemcpy3D(&parms);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1)));
  checkError(cudaMemcpy3D(&parms));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol("constData", h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device));
  err = cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(constData.get_ptr(), h_A, size, dpct::host_to_device)));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyHostToDevice));

  // CHECK:  dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK:  dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol("constData", h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device));
  err = cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::host_to_device)));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyHostToDevice));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::device_to_device);
  cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::device_to_device);
  cudaMemcpyToSymbol("constData", d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::device_to_device));
  err = cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(constData.get_ptr(), h_A, size, dpct::device_to_device)));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 0, cudaMemcpyDeviceToDevice));

  // CHECK:  dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::device_to_device);
  cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:  dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::device_to_device);
  cudaMemcpyToSymbol("constData", d_B, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::device_to_device));
  err = cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, h_A, size, dpct::device_to_device)));
  checkError(cudaMemcpyToSymbol(constData, h_A, size, 1, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::automatic);
  cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:   err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::automatic));
  err = cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(constData.get_ptr(), d_B, size, dpct::automatic)));
  checkError(cudaMemcpyToSymbol(constData, d_B, size, 0, cudaMemcpyDefault));

  // CHECK:  dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::automatic);
  cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDefault);
  // CHECK:   err = CHECK_SYCL_ERROR(dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::automatic));
  err = cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDefault);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy((char *)(constData.get_ptr()) + 1, d_B, size, dpct::automatic)));
  checkError(cudaMemcpyToSymbol(constData, d_B, size, 1, cudaMemcpyDefault));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), d_B, size);
  cudaMemcpyToSymbol(constData, d_B, size);
  // CHECK:  dpct::dpct_memcpy(constData.get_ptr(), d_B, size);
  cudaMemcpyToSymbol("constData", d_B, size);
  // CHECK:   err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(constData.get_ptr(), d_B, size));
  err = cudaMemcpyToSymbol(constData, d_B, size);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(constData.get_ptr(), d_B, size)));
  checkError(cudaMemcpyToSymbol(constData, d_B, size));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, constData, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:  dpct::dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, "constData", size, 0, cudaMemcpyDeviceToHost);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host));
  err = cudaMemcpyFromSymbol(h_A, constData, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, constData.get_ptr(), size, dpct::device_to_host)));
  checkError(cudaMemcpyFromSymbol(h_A, constData, size, 0, cudaMemcpyDeviceToHost));

  // CHECK:  dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host));
  err = cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_host)));
  checkError(cudaMemcpyFromSymbol(h_A, constData, size, 1, cudaMemcpyDeviceToHost));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(d_B, constData.get_ptr(), size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_B, constData, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_B, constData.get_ptr(), size, dpct::device_to_device));
  err = cudaMemcpyFromSymbol(d_B, constData, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_B, constData.get_ptr(), size, dpct::device_to_device)));
  checkError(cudaMemcpyFromSymbol(d_B, constData, size, 0, cudaMemcpyDeviceToDevice));


  // CHECK:  dpct::dpct_memcpy(d_B, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_B, constData, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:  dpct::dpct_memcpy(d_B, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_B, "constData", size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_B, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_device));
  err = cudaMemcpyFromSymbol(d_B, constData, size, 1, cudaMemcpyDeviceToDevice);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_B, (char *)(constData.get_ptr()) + 1, size, dpct::device_to_device)));
  checkError(cudaMemcpyFromSymbol(d_B, constData, size, 1, cudaMemcpyDeviceToDevice));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK:  dpct::dpct_memcpy(h_A, constData.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, "constData", size);
  // CHECK:   err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, constData.get_ptr(), size));
  err = cudaMemcpyFromSymbol(h_A, constData, size);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, constData.get_ptr(), size)));
  checkError(cudaMemcpyFromSymbol(h_A, constData, size));

  // CHECK: dpct::dpct_free(d_A);
  cudaFree(d_A);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_free(d_A));
  err = cudaFree(d_A);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_free(d_A)));
  checkError(cudaFree(d_A));
  // CHECK:  free(h_A);
  free(h_A);
}

// CHECK: static dpct::global_memory<float, 1> d_A(1234567);
// CHECK: static dpct::global_memory<float, 1> d_B(1234567);
static __device__ float d_A[1234567];
static __device__ float d_B[1234567];

void testCommas_in_global_memory() {
  size_t size = 1234567 * sizeof(float);
  cudaError_t err;
  float *h_A = (float *)malloc(size);

  // CHECK:  dpct::dpct_memset(d_A.get_ptr(), 0xf, size);
  cudaMemset(d_A, 0xf, size);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memset(d_A.get_ptr(), 0xf, size));
  err = cudaMemset(d_A, 0xf, size);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memset(d_A.get_ptr(), 0xf, size)));
  checkError(cudaMemset(d_A, 0xf, size));

  ///////// Host to host
  // CHECK:  dpct::dpct_memcpy(h_A, h_A, size, dpct::host_to_host);
  cudaMemcpy(h_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, h_A, size, dpct::host_to_host));
  err = cudaMemcpy(h_A, h_A, size, cudaMemcpyHostToHost);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, h_A, size, dpct::host_to_host)));
  checkError(cudaMemcpy(h_A, h_A, size, cudaMemcpyHostToHost));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device));
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device)));
  checkError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host));
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host)));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

  ///////// Device to Device
  // CHECK:  dpct::dpct_memcpy(d_B.get_ptr(), d_A.get_ptr(), size, dpct::device_to_device);
  cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_B.get_ptr(), d_A.get_ptr(), size, dpct::device_to_device));
  err = cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_B.get_ptr(), d_A.get_ptr(), size, dpct::device_to_device)));
  checkError(cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::automatic);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::automatic));
  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::automatic)));
  checkError(cudaMemcpy(h_A, d_A, size, cudaMemcpyDefault));

  ///////// Host to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device);
  cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device));
  err = cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A.get_ptr(), h_A, size, dpct::host_to_device)));
  checkError(cudaMemcpyToSymbol(d_A, h_A, size, 0, cudaMemcpyHostToDevice));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device);
  cudaMemcpyToSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  err = cudaMemcpyToSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device)));
  checkError(cudaMemcpyToSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice));

  ///////// Default
  // CHECK:  dpct::dpct_memcpy(h_A, d_B.get_ptr(), size, dpct::automatic);
  cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:   err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_B.get_ptr(), size, dpct::automatic));
  err = cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_B.get_ptr(), size, dpct::automatic)));
  checkError(cudaMemcpyToSymbol(h_A, d_B, size, 0, cudaMemcpyDefault));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy(h_A, d_B.get_ptr(), size);
  cudaMemcpyToSymbol(h_A, d_B, size);
  // CHECK:   err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_B.get_ptr(), size));
  err = cudaMemcpyToSymbol(h_A, d_B, size);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_B.get_ptr(), size)));
  checkError(cudaMemcpyToSymbol(h_A, d_B, size));

  ///////// Device to host
  // CHECK:  dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host);
  cudaMemcpyFromSymbol(h_A, d_A, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host));
  err = cudaMemcpyFromSymbol(h_A, d_A, size, 0, cudaMemcpyDeviceToHost);
  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_A.get_ptr(), size, dpct::device_to_host)));
  checkError(cudaMemcpyFromSymbol(h_A, d_A, size, 0, cudaMemcpyDeviceToHost));

  ///////// Device to device
  // CHECK:  dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device);
  cudaMemcpyFromSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:  err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device));
  err = cudaMemcpyFromSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(d_A.get_ptr(), d_B.get_ptr(), size, dpct::device_to_device)));
  checkError(cudaMemcpyFromSymbol(d_A, d_B, size, 0, cudaMemcpyDeviceToDevice));

  ///////// Default parameter overload
  // CHECK:  dpct::dpct_memcpy(h_A, d_B.get_ptr(), size);
  cudaMemcpyFromSymbol(h_A, d_B, size);
  // CHECK:   err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_B.get_ptr(), size));
  err = cudaMemcpyFromSymbol(h_A, d_B, size);
  // CHECK:   checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(h_A, d_B.get_ptr(), size)));
  checkError(cudaMemcpyFromSymbol(h_A, d_B, size));

  void *p_addr;
  // CHECK:  *(&p_addr) = d_A.get_ptr();
  cudaGetSymbolAddress(&p_addr, d_A);

  // CHECK:  err = CHECK_SYCL_ERROR(*(&p_addr) = d_A.get_ptr());
  err = cudaGetSymbolAddress(&p_addr, d_A);

  // CHECK:checkError(CHECK_SYCL_ERROR(*(&p_addr) = d_A.get_ptr()));
  checkError(cudaGetSymbolAddress(&p_addr, d_A));

  size_t size2;
  // CHECK: size2 = d_A.get_size();
  cudaGetSymbolSize(&size2, d_A);

  // CHECK:  err = CHECK_SYCL_ERROR(size2 = d_A.get_size());
  err = cudaGetSymbolSize(&size2, d_A);

  // CHECK:checkError(CHECK_SYCL_ERROR(size2 = d_A.get_size()));
  checkError(cudaGetSymbolSize(&size2, d_A));

  size_t result1, result2;
  // CHECK: /*
  // CHECK: DPCT1072:{{[0-9]+}}: SYCL currently does not support getting the available memory on the current device. You may need to adjust the code.
  // CHECK: */
  // CHECK: result2 = dpct::get_current_device().get_device_info().get_global_mem_size();
  cuMemGetInfo(&result1, &result2);

  CUresult cu_err;
  // CHECK: /*
  // CHECK: DPCT1072:{{[0-9]+}}: SYCL currently does not support getting the available memory on the current device. You may need to adjust the code.
  // CHECK: */
  // CHECK: cu_err = CHECK_SYCL_ERROR(result2 = dpct::get_current_device().get_device_info().get_global_mem_size());
  cu_err = cuMemGetInfo(&result1, &result2);

  // CHECK: /*
  // CHECK: DPCT1072:{{[0-9]+}}: SYCL currently does not support getting the available memory on the current device. You may need to adjust the code.
  // CHECK: */
  // CHECK: cuCheckError(CHECK_SYCL_ERROR(result2 = dpct::get_current_device().get_device_info().get_global_mem_size()));
  cuCheckError(cuMemGetInfo(&result1, &result2));

  // CHECK: /*
  // CHECK: DPCT1072:{{[0-9]+}}: SYCL currently does not support getting the available memory on the current device. You may need to adjust the code.
  // CHECK: */
  // CHECK: result2 = dpct::get_current_device().get_device_info().get_global_mem_size();
  cudaMemGetInfo(&result1, &result2);


  // CHECK: /*
  // CHECK: DPCT1072:{{[0-9]+}}: SYCL currently does not support getting the available memory on the current device. You may need to adjust the code.
  // CHECK: */
  // CHECK: err = CHECK_SYCL_ERROR(result2 = dpct::get_current_device().get_device_info().get_global_mem_size());
  err = cudaMemGetInfo(&result1, &result2);

  // CHECK: /*
  // CHECK: DPCT1072:{{[0-9]+}}: SYCL currently does not support getting the available memory on the current device. You may need to adjust the code.
  // CHECK: */
  // CHECK: checkError(CHECK_SYCL_ERROR(result2 = dpct::get_current_device().get_device_info().get_global_mem_size()));
  checkError(cudaMemGetInfo(&result1, &result2));

  CUdeviceptr  devicePtr;
  // CHECK: devicePtr = (dpct::device_ptr)dpct::dpct_malloc(size, size, size);
  cuMemAllocPitch((CUdeviceptr *)&devicePtr, &size, size, size, size);

  // CHECK: cu_err = CHECK_SYCL_ERROR(devicePtr = (dpct::device_ptr)dpct::dpct_malloc(size, size, size));
  cu_err = cuMemAllocPitch((CUdeviceptr *)&devicePtr, &size, size, size, size);
  // CHECK:  cuCheckError(CHECK_SYCL_ERROR(devicePtr = (dpct::device_ptr)dpct::dpct_malloc(size, size, size)));
  cuCheckError(cuMemAllocPitch((CUdeviceptr *)&devicePtr, &size, size, size, size));

  int* a;
  cudaStream_t stream;
  int deviceID = 0;
  CUdevice cudevice =0;
  CUdeviceptr devPtr;
  // CHECK:/*
  // CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaMemPrefetchAsync is not supported.
  // CHECK-NEXT:*/
  cudaMemPrefetchAsync (a, 100, deviceID, stream);

  // CHECK:/*
  // CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cuMemPrefetchAsync is not supported.
  // CHECK-NEXT:*/
  cuMemPrefetchAsync(devPtr, 100, cudevice, stream);

  // CHECK:  free(h_A);
  free(h_A);
}

#define MY_CHECKER(CALL)                                                           \
    if ((CALL) != cudaSuccess) { \
        exit(-1); \
    }

#define MY_ERROR_CHECKER(CALL) my_error_checker((CALL), #CALL)
template <typename T>
void my_error_checker(T ReturnValue, char const *const FuncName) {}


template<typename T>
void uninstantiated_template_call(const T * d_data, size_t width, size_t height) {
  size_t datasize = width * height;
  T * data = new T[datasize];
  cudaMemcpy3DParms parms;
  // CHECK:  assert_cuda(CHECK_SYCL_ERROR(dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host)));
  assert_cuda(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK: dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host);
  cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

#define DATAMACRO data+32*32

  // CHECK: dpct::dpct_memcpy(DATAMACRO, d_data, datasize * sizeof(T), dpct::device_to_host);
  cudaMemcpy(DATAMACRO, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(32*32+DATAMACRO, d_data, datasize * sizeof(T), dpct::device_to_host);
  cudaMemcpy(32*32+DATAMACRO, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK:  checkError(CHECK_SYCL_ERROR(dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host)));
  checkError(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK: dpct::err0 err = CHECK_SYCL_ERROR(dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host));
  cudaError_t err = cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK: MY_CHECKER(CHECK_SYCL_ERROR(dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host)));
  MY_CHECKER(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK: MY_ERROR_CHECKER(CHECK_SYCL_ERROR(dpct::dpct_memcpy(data, d_data, datasize * sizeof(T), dpct::device_to_host)));
  MY_ERROR_CHECKER(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

  // CHECK: #define CUDAMEMCPY dpct::dpct_memcpy
  // CHECK-NEXT: CUDAMEMCPY(data, d_data, datasize * sizeof(T), dpct::device_to_host);
  #define CUDAMEMCPY cudaMemcpy
  CUDAMEMCPY(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost);

  // CHECK:  assert_cuda(CHECK_SYCL_ERROR(dpct::dpct_memcpy(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host)));
  assert_cuda(cudaMemcpy2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost));

  // CHECK: dpct::dpct_memcpy(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host);
  cudaMemcpy2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(DATAMACRO, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host);
  cudaMemcpy2D(DATAMACRO, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost);

  // CHECK: dpct::dpct_memcpy(32*32+DATAMACRO, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host);
  cudaMemcpy2D(32*32+DATAMACRO, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost);

  // CHECK: MY_CHECKER(CHECK_SYCL_ERROR(dpct::dpct_memcpy(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host)));
  MY_CHECKER(cudaMemcpy2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost));

  // CHECK: MY_ERROR_CHECKER(CHECK_SYCL_ERROR(dpct::dpct_memcpy(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host)));
  MY_ERROR_CHECKER(cudaMemcpy2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost));

  // CHECK: #define CUDAMEMCPY2D dpct::dpct_memcpy
  // CHECK-NEXT: CUDAMEMCPY2D(data, datasize, d_data, datasize, datasize, datasize, dpct::device_to_host);
  #define CUDAMEMCPY2D cudaMemcpy2D
  CUDAMEMCPY2D(data, datasize, d_data, datasize, datasize, datasize, cudaMemcpyDeviceToHost);

  // CHECK: MY_CHECKER(CHECK_SYCL_ERROR(dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1)));
  MY_CHECKER(cudaMemcpy3D(&parms));

  // CHECK: MY_ERROR_CHECKER(CHECK_SYCL_ERROR(dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1)));
  MY_ERROR_CHECKER(cudaMemcpy3D(&parms));

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
  */
  cudaMalloc(&buffer, N*sizeof(float));
}

// CHECK: static dpct::global_memory<uint32_t, 1> d_error(1);
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

  //CHECK: checkError(CHECK_SYCL_ERROR([&](){
  //CHECK-NEXT:   desc = array->get_channel();
  //CHECK-NEXT:   extent = array->get_range();
  //CHECK-NEXT:   flags = 0;
  //CHECK-NEXT:   }()));
  checkError(cudaArrayGetInfo(&desc, &extent, &flags, array));

  //CHECK: errorCode = CHECK_SYCL_ERROR([&](){
  //CHECK-NEXT:   desc = array->get_channel();
  //CHECK-NEXT:   extent = array->get_range();
  //CHECK-NEXT:   flags = 0;
  //CHECK-NEXT:   }());
  errorCode = cudaArrayGetInfo(&desc, &extent, &flags, array);

  int host;
  // CHECK: flags = 0;
  cudaHostGetFlags(&flags, &host);

  // CHECK: checkError(CHECK_SYCL_ERROR(flags = 0));
  checkError(cudaHostGetFlags(&flags, &host));

  // CHECK: errorCode = CHECK_SYCL_ERROR(flags = 0);
  errorCode = cudaHostGetFlags(&flags, &host);

  int *devPtr;

  CUdeviceptr devicePtr;

  CUresult cu_err;

  CUdeviceptr cuDevPtr;

  CUdevice cudevice =0;

  CUmem_advise advise = CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION;

  size_t count;
  // CHECK: int advice;
  cudaMemoryAdvise advice;
  int device;

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaMemAdvise is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaMemAdvise(devPtr, count, advice, device);
  cudaMemAdvise(devPtr, count, advice, device);

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaMemAdvise is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError(cudaMemAdvise(devPtr, count, advice, device));
  checkError(cudaMemAdvise(devPtr, count, advice, device));

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaMemAdvise is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = cudaMemAdvise(devPtr, count, advice, device);
  errorCode = cudaMemAdvise(devPtr, count, advice, device);

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaMemAdvise is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
  // CHECK-NEXT: */
  // CHECK-NEXT: cudaMemAdvise(devPtr, count, 0, device);
  cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device);

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaMemAdvise is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
  // CHECK-NEXT: */
  // CHECK-NEXT: checkError(cudaMemAdvise(devPtr, count, 0, device));
  checkError(cudaMemAdvise(devPtr, count, cudaMemAdviseSetReadMostly, device));

  // CHECK: /*
  // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cudaMemAdvise is not supported.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
  // CHECK-NEXT: */
  // CHECK-NEXT: errorCode = cudaMemAdvise(devPtr, count, 0, device);
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

