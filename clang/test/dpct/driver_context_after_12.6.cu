// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3, v12.4
// RUN: dpct --format-range=none -out-root %T/driver_context_after_12.6 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/driver_context_after_12.6/driver_context_after_12.6.dp.cpp
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/driver_context_after_12.6/driver_context_after_12.6.dp.cpp -o %T/driver_context_after_12.6/driver_context_after_12.6.dp.o %}
#include <cuda.h>
#include <cuda_runtime_api.h>

int main(){
#ifndef NO_BUILD_TEST
  CUdevice device;

  // CHECK: int ctx;
  CUcontext ctx;

  unsigned int flags = CU_CTX_MAP_HOST;
  CUexecAffinityParam* paramsArray;
  // CHECK: ctx = dpct::push_device_for_curr_thread(device);
  cuCtxCreate_v3(&ctx, paramsArray, 1, flags, device);

  CUctxCreateParams* ctxCreateParams;
  // CHECK: ctx = dpct::push_device_for_curr_thread(device);
  cuCtxCreate_v4(&ctx, ctxCreateParams, flags, device);
#endif
  return 0;
}
