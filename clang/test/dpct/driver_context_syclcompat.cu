// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3, v12.4
// RUN: dpct --format-range=none -out-root %T/driver_context_syclcompat %s --use-syclcompat --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/driver_context_syclcompat/driver_context_syclcompat.dp.cpp
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/driver_context_syclcompat/driver_context_syclcompat.dp.cpp -o %T/driver_context_syclcompat/driver_context_syclcompat.dp.o %}
#include <cuda.h>
#include <cuda_runtime_api.h>

#define MY_SAFE_CALL(CALL) do {    \
  int Error = CALL;                \
} while (0)

int main(){
#ifndef NO_BUILD_TEST
  CUdevice device;

  // CHECK: int ctx;
  CUcontext ctx;

  unsigned int flags = CU_CTX_MAP_HOST;
  CUexecAffinityParam* paramsArray;
  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cuCtxCreate_v3" is not currently supported with SYCLcompat. Please adjust the code manually.
  cuCtxCreate_v3(&ctx, paramsArray, 1, flags, device);

  CUctxCreateParams* ctxCreateParams;
  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cuCtxCreate_v4" is not currently supported with SYCLcompat. Please adjust the code manually.
  cuCtxCreate_v4(&ctx, ctxCreateParams, flags, device);  

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cuCtxCreate_v2" is not currently supported with SYCLcompat. Please adjust the code manually.
  cuCtxCreate(&ctx, CU_CTX_LMEM_RESIZE_TO_MAX, device);

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cuCtxPushCurrent_v2" is not currently supported with SYCLcompat. Please adjust the code manually.
  MY_SAFE_CALL(cuCtxPushCurrent(ctx));

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cuCtxPopCurrent_v2" is not currently supported with SYCLcompat. Please adjust the code manually.
  cuCtxPopCurrent(&ctx);

  // CHECK: ctx = syclcompat::select_device(device);
  cuDevicePrimaryCtxRetain(&ctx, device);

  // CHECK: MY_SAFE_CALL(SYCLCOMPAT_CHECK_ERROR(syclcompat::select_device(ctx)));
  MY_SAFE_CALL(cuCtxSetCurrent(ctx));
#endif
  return 0;
}

#ifndef NO_BUILD_TEST
void foo() {
  float *h_A;
  unsigned int numAttributes = 5;

  CUpointer_attribute attributes[] = {
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
    CU_POINTER_ATTRIBUTE_HOST_POINTER,
    CU_POINTER_ATTRIBUTE_IS_MANAGED,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
  };

  CUmemorytype memType;
  void* hostPtr;
  unsigned int isManaged;
  int deviceID;
  CUdeviceptr devPtr;

  void* attributeValues[] = {
    &memType,
    &devPtr,
    &hostPtr,
    &isManaged,
    &deviceID
  };

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cuPointerGetAttributes" is not currently supported with SYCLcompat. Please adjust the code manually.
  cuPointerGetAttributes(
    numAttributes,
    attributes,
    attributeValues,
    (CUdeviceptr) h_A
  );
}
#endif
