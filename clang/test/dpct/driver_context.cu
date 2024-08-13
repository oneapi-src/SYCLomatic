// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v12.0, v12.1, v12.2, v12.3, v12.4
// RUN: dpct --format-range=none -out-root %T/driver_context %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/driver_context/driver_context.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/driver_context/driver_context.dp.cpp -o %T/driver_context/driver_context.dp.o %}
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
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuInit was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  cuInit(0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuInit was replaced with 0 because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_SAFE_CALL(0);
  MY_SAFE_CALL(cuInit(0));

  // CHECK: ctx = dpct::push_device_for_curr_thread(device);
  cuCtxCreate(&ctx, CU_CTX_LMEM_RESIZE_TO_MAX, device);

  unsigned int flags = CU_CTX_MAP_HOST;
  CUexecAffinityParam* paramsArray;
  // CHECK: ctx = dpct::select_device(device);
  cuCtxCreate_v3(&ctx, paramsArray, 1, flags, device);

  CUctxCreateParams* ctxCreateParams;
  // CHECK: ctx = dpct::select_device(device);
  cuCtxCreate_v4(&ctx, ctxCreateParams, flags, device);


  // CHECK: ctx = dpct::select_device(device);
  cuDevicePrimaryCtxRetain(&ctx, device);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuDevicePrimaryCtxRelease_v2 was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  cuDevicePrimaryCtxRelease(device);

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(ctx = dpct::push_device_for_curr_thread(device)));
  MY_SAFE_CALL(cuCtxCreate(&ctx, CU_CTX_LMEM_RESIZE_TO_MAX, device));

  // CHECK: dpct::select_device(ctx);
  cuCtxSetCurrent(ctx);

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(dpct::select_device(ctx)));
  MY_SAFE_CALL(cuCtxSetCurrent(ctx));

  // CHECK: dpct::get_current_device().ext_oneapi_enable_peer_access(dpct::dev_mgr::instance().get_device(ctx2));
  cuCtxEnablePeerAccess(ctx2, 0);

  // CHECK: ctx2 = dpct::dev_mgr::instance().current_device_id();
  cuCtxGetCurrent(&ctx2);

  // CHECK: if (ctx == -1) {
  if (ctx == nullptr) {
    return 0;
  }

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(ctx2 = dpct::dev_mgr::instance().current_device_id()));
  MY_SAFE_CALL(cuCtxGetCurrent(&ctx2));

  // CHECK: dpct::push_device_for_curr_thread(ctx);
  cuCtxPushCurrent(ctx);

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(dpct::push_device_for_curr_thread(ctx)));
  MY_SAFE_CALL(cuCtxPushCurrent(ctx));

  CUcontext *ctx_ptr;
  // CHECK: *ctx_ptr = dpct::pop_device_for_curr_thread();
  cuCtxPopCurrent(ctx_ptr);

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(ctx = dpct::pop_device_for_curr_thread()));
  MY_SAFE_CALL(cuCtxPopCurrent(&ctx));

  // CHECK: dpct::get_current_device().queues_wait_and_throw();
  cuCtxSynchronize();

  // CHECK: MY_SAFE_CALL(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  MY_SAFE_CALL(cuCtxSynchronize());

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuCtxDestroy_v2 was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  cuCtxDestroy(ctx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuCtxDestroy_v2 was replaced with 0 because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: MY_SAFE_CALL(0);
  MY_SAFE_CALL(cuCtxDestroy(ctx2));

  // CHECK: int* dev_ptr;
  // CHECK-NEXT: *dev_ptr = dpct::get_current_device_id();
  CUdevice* dev_ptr;
  cuCtxGetDevice(dev_ptr);

  unsigned int ver;
  // CHECK: ver = dpct::get_sycl_language_version();
  cuCtxGetApiVersion(ctx, &ver);

  return 0;
}
