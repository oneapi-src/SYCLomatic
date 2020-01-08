// RUN: cat %s > %T/cuda-device-api.cu
// RUN: cd %T
// RUN: dpct -out-root %T cuda-device-api.cu --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-device-api.dp.cpp --match-full-lines cuda-device-api.cu

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
  // CHECK-NEXT: DPCT1029:0: DPC++ currently doesn't support getting limits on devices; the
  // CHECK-NEXT: output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: *pValue = 0;
  cudaDeviceGetLimit(pValue, limit);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceSetSharedMemConfig was removed, because
  // CHECK-NEXT: DPC++ currently doesn't support configuring shared memory on devices.
  // CHECK-NEXT: */
  cudaDeviceSetSharedMemConfig(config);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaSetDeviceFlags was removed, because DPC++
  // CHECK-NEXT: currently doesn't support setting flags for devices.
  // CHECK-NEXT: */
  cudaSetDeviceFlags(flags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceEnablePeerAccess was removed, because DPC++
  // CHECK-NEXT: currently doesn't support memory access across peer devices.
  // CHECK-NEXT: */
  cudaDeviceEnablePeerAccess(peerDevice, flags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaDeviceDisablePeerAccess was removed, because DPC++
  // CHECK-NEXT: currently doesn't support memory access across peer devices.
  // CHECK-NEXT: */
  cudaDeviceDisablePeerAccess(peerDevice);

  // CHECK: /*
  // CHECK-NEXT: DPCT1031:{{[0-9]+}}: DPC++ currently doesn't support memory access across peer devices;
  // CHECK-NEXT: the output parameter(s) are set to 0.
  // CHECK-NEXT: */
  // CHECK-NEXT: *canAccessPeer = 0;
  cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: DPC++ currently doesn't support IPC operations. You may need to
  // CHECK-NEXT: rewrite the code.
  // CHECK-NEXT: */
  cudaIpcGetEventHandle(handleEvent, event);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: DPC++ currently doesn't support IPC operations. You may need to
  // CHECK-NEXT: rewrite the code.
  // CHECK-NEXT: */
  cudaIpcOpenEventHandle(&event, *handleEvent);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: DPC++ currently doesn't support IPC operations. You may need to
  // CHECK-NEXT: rewrite the code.
  // CHECK-NEXT: */
  cudaIpcGetMemHandle(handleMem, devPtr);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: DPC++ currently doesn't support IPC operations. You may need to
  // CHECK-NEXT: rewrite the code.
  // CHECK-NEXT: */
  cudaIpcOpenMemHandle(&devPtr, *handleMem, flags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1030:{{[0-9]+}}: DPC++ currently doesn't support IPC operations. You may need to
  // CHECK-NEXT: rewrite the code.
  // CHECK-NEXT: */
  cudaIpcCloseMemHandle(devPtr);
}
