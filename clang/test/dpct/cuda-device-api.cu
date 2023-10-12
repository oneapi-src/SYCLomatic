// RUN: cat %s > %T/cuda-device-api.cu
// RUN: cd %T
// RUN: dpct -out-root %T/cuda-device-api cuda-device-api.cu --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-device-api/cuda-device-api.dp.cpp --match-full-lines cuda-device-api.cu

void foo() {
  size_t *pValue;
  cudaLimit limit;
  cudaSharedMemConfig config;
  unsigned flags;
  int peerDevice;
  int *canAccessPeer;
  int device;
  cudaIpcEventHandle_t *handleEvent;
  cudaEvent_t event;
  cudaIpcMemHandle_t *handleMem;
  void *devPtr;

  // CHECK: /*
  // CHECK-NEXT: DPCT1029:0: SYCL currently does not support getting device resource limits.
  // CHECK-NEXT: The output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: *pValue = 0;
  cudaDeviceGetLimit(pValue, limit);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceSetSharedMemConfig was removed because SYCL
  // CHECK-NEXT: currently does not support configuring shared memory on devices.
  // CHECK-NEXT:*/
  cudaDeviceSetSharedMemConfig(config);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaSetDeviceFlags was removed because SYCL currently
  // CHECK-NEXT: does not support setting flags for devices.
  // CHECK-NEXT: */
  cudaSetDeviceFlags(flags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceEnablePeerAccess was removed because SYCL
  // CHECK-NEXT: currently does not support memory access across peer devices.
  // CHECK-NEXT: */
  cudaDeviceEnablePeerAccess(peerDevice, flags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceDisablePeerAccess was removed because SYCL
  // CHECK-NEXT: currently does not support memory access across peer devices.
  // CHECK-NEXT: */
  cudaDeviceDisablePeerAccess(peerDevice);

  // CHECK:      /*
  // CHECK-NEXT: DPCT1031:{{[0-9]+}}: Memory accessing across peer devices is a implementation-specific
  // CHECK-NEXT: feature which may not be supported by all SYCL backends and compilers. The
  // CHECK-NEXT: output parameter(s) are set to 0.
  // CHECK-NEXT: */

  // CHECK-NEXT: *canAccessPeer = 0;
  cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: SYCL currently does not support inter-process communication (IPC)
  // CHECK-NEXT: operations. You may need to rewrite the code.
  // CHECK-NEXT: */

  cudaIpcGetEventHandle(handleEvent, event);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: SYCL currently does not support inter-process communication (IPC)
  // CHECK-NEXT: operations. You may need to rewrite the code.
  // CHECK-NEXT: */
  cudaIpcOpenEventHandle(&event, *handleEvent);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: SYCL currently does not support inter-process communication (IPC)
  // CHECK-NEXT: operations. You may need to rewrite the code.
  // CHECK-NEXT: */
  cudaIpcGetMemHandle(handleMem, devPtr);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: SYCL currently does not support inter-process communication (IPC)
  // CHECK-NEXT: operations. You may need to rewrite the code.
  // CHECK-NEXT: */
  cudaIpcOpenMemHandle(&devPtr, *handleMem, flags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: SYCL currently does not support inter-process communication (IPC)
  // CHECK-NEXT: operations. You may need to rewrite the code.
  // CHECK-NEXT: */
  cudaIpcCloseMemHandle(devPtr);


  cudaFuncCache fconfig;

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaFuncSetSharedMemConfig was removed because SYCL
  // CHECK-NEXT: currently does not support configuring shared memory on devices.
  cudaFuncSetSharedMemConfig(NULL, config );

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaFuncSetCacheConfig was removed because SYCL
  // CHECK-NEXT: currently does not support configuring shared memory on devices.
  cudaFuncSetCacheConfig(NULL, fconfig);

}
