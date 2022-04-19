// RUN: c2s --format-range=none -out-root %T/driver_context %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/driver_context/driver_context.dp.cpp
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NUM 1
#define MY_SAFE_CALL(CALL) do {    \
  int Error = CALL;                \
} while (0)

int main(){

  CUdevice device;

  // CHECK: int ctx;
  CUcontext ctx;

  // CHECK: int ctx2;
  CUcontext ctx2;

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuInit was removed because the function call is redundant in DPC++.
  // CHECK-NEXT: */
  cuInit(0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuInit was replaced with 0 because the function call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_SAFE_CALL(0);
  MY_SAFE_CALL(cuInit(0));

  // CHECK: ctx = device;
  cuCtxCreate(&ctx, CU_CTX_LMEM_RESIZE_TO_MAX, device);

  // CHECK: MY_SAFE_CALL((ctx = device, 0));
  MY_SAFE_CALL(cuCtxCreate(&ctx, CU_CTX_LMEM_RESIZE_TO_MAX, device));

  // CHECK: c2s::dev_mgr::instance().select_device(ctx);
  cuCtxSetCurrent(ctx);

  // CHECK: MY_SAFE_CALL((c2s::dev_mgr::instance().select_device(ctx), 0));
  MY_SAFE_CALL(cuCtxSetCurrent(ctx));

  // CHECK: ctx2 = c2s::dev_mgr::instance().current_device_id();
  cuCtxGetCurrent(&ctx2);

  // CHECK: MY_SAFE_CALL((ctx2 = c2s::dev_mgr::instance().current_device_id(), 0));
  MY_SAFE_CALL(cuCtxGetCurrent(&ctx2));

  // CHECK: c2s::get_current_device().queues_wait_and_throw();
  cuCtxSynchronize();

  // CHECK: MY_SAFE_CALL((c2s::get_current_device().queues_wait_and_throw(), 0));
  MY_SAFE_CALL(cuCtxSynchronize());

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuCtxDestroy_v2 was removed because the function call is redundant in DPC++.
  // CHECK-NEXT: */
  cuCtxDestroy(ctx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuCtxDestroy_v2 was replaced with 0 because the function call is redundant in DPC++.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_SAFE_CALL(0);
  MY_SAFE_CALL(cuCtxDestroy(ctx2));

  return 0;
}
