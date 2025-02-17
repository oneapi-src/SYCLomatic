// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -out-root %T/cuGraphics_default_option %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuGraphics_default_option/cuGraphics_default_option.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/cuGraphics_default_option/cuGraphics_default_option.dp.cpp -o %T/cuGraphics_default_option/cudaGraphicsResource_test.dp.o %}

#ifndef NO_BUILD_TEST
#include <cuda.h>

int main() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of CUgraphicsResource is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  CUgraphicsResource resource, *resources;

  CUdeviceptr devPtr;
  size_t size;

  CUstream stream;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cuGraphicsMapResources is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cuGraphicsMapResources(2, resources, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cuGraphicsMapResources is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cuGraphicsMapResources(1, &resource, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cuGraphicsResourceGetMappedPointer_v2 is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cuGraphicsResourceGetMappedPointer(&devPtr, &size, resource);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cuGraphicsUnmapResources is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cuGraphicsUnmapResources(2, resources, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cuGraphicsUnmapResources is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cuGraphicsUnmapResources(1, &resource, stream);

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cuGraphicsUnregisterResource is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cuGraphicsUnregisterResource(resource);

  return 0;
}

#endif
