// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -out-root %T/cudaGraphics_default_option %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaGraphics_default_option/cudaGraphics_default_option.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DBUILD_TEST -fsycl %T/cudaGraphics_default_option/cudaGraphics_default_option.dp.cpp -o %T/cudaGraphics_default_option/cudaGraphicsResource_test.dp.o %}

#ifndef BUILD_TEST
#include <cuda.h>

int main() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsResource_t is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsResource_t resource, *resources;

  void *devPtr;
  cudaMipmappedArray_t* mipmappedArray;
  cudaArray_t* array;
  size_t size;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsRegisterFlags is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsRegisterFlagsNone is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsRegisterFlags regFlags = cudaGraphicsRegisterFlagsNone;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsMapFlags is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsMapFlagsNone is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsMapFlags mapFlags = cudaGraphicsMapFlagsNone;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaGraphicsResourceSetMapFlags was removed because this functionality is deprecated in DX12 and hence is not supported in SYCL.
  // CHECK-NEXT: */
  cudaGraphicsResourceSetMapFlags(resource, mapFlags);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsMapResources is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsMapResources(1, resources);

  cudaStream_t stream;
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsMapResources is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsMapResources(1, resources, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsResourceGetMappedPointer is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsResourceGetMappedPointer(&devPtr, &size, resource);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsResourceGetMappedMipmappedArray is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);

  unsigned int arrayIndex, mipLevel;
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsSubResourceGetMappedArray is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsUnmapResources is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsUnmapResources(1, resources);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsUnmapResources is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsUnmapResources(1, resources, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsUnregisterResource is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsUnregisterResource(resource);

  return 0;
}

#endif
