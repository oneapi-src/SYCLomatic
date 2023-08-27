// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/cache_remove %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cache_remove/cache_remove.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>

__global__ void kernel()
{
  return;
}

int main()
{
  cudaStream_t stream;
  const int pre_transform_tensor_size =
      8 * 8 * 8;
  const int transformed_tensor_size = pre_transform_tensor_size * 36 / 16;
  const int res_block_mem =
      transformed_tensor_size * 2 + pre_transform_tensor_size;
  float *aptr;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cudaLaunchAttributeValue is not supported.
  cudaStreamAttrValue stream_attribute = {};
  // CHECK: // begin for check remove stream_attribute.accessPolicyWindow
  // CHECK-NEXT: // end for check remove stream_attribute.accessPolicyWindow
  // begin for check remove stream_attribute.accessPolicyWindow
  stream_attribute.accessPolicyWindow.base_ptr = aptr;
  stream_attribute.accessPolicyWindow.num_bytes = res_block_mem;
  stream_attribute.accessPolicyWindow.hitRatio = 1.0f;
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  // end for check remove stream_attribute.accessPolicyWindow
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaStreamSetAttribute was removed because this call is redundant in SYCL.
  cudaStreamSetAttribute(
      stream, cudaLaunchAttributeIgnore, &stream_attribute);
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaStreamSetAttribute was removed because SYCL currently does not support setting cache config on devices.
  cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaStreamGetAttribute was removed because this call is redundant in SYCL.
  cudaStreamGetAttribute(
      stream, cudaLaunchAttributeIgnore, &stream_attribute);
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaStreamGetAttribute was removed because SYCL currently does not support setting cache config on devices.
  cudaStreamGetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaCtxResetPersistingL2Cache was removed because SYCL currently does not support setting cache config on devices.
  cudaCtxResetPersistingL2Cache();
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cuCtxResetPersistingL2Cache was removed because SYCL currently does not support setting cache config on devices.
  cuCtxResetPersistingL2Cache();
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaFuncSetAttribute was removed because SYCL currently does not support corresponding setting.
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 8 * 8);
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaFuncSetAttribute was removed because SYCL currently does not support corresponding setting.
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 8 * 8 * 8);
  // CHECK: DPCT1026:{{[0-9]+}}: The call to cudaFuncSetAttribute was removed because SYCL currently does not support corresponding setting.
  cudaFuncSetAttribute(kernel, cudaFuncAttributeClusterDimMustBeSet, 8 * 8 * 8);
}
