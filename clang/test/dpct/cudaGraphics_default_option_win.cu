// UNSUPPORTED: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none -out-root %T/cudaGraphics_default_option_win %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaGraphics_default_option_win/cudaGraphics_default_option_win.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/cudaGraphics_default_option_win/cudaGraphics_default_option_win.dp.cpp -o %T/cudaGraphics_default_option_win/cudaGraphicsResource_test.dp.o %}

#ifndef NO_BUILD_TEST
#include <cuda.h>
#include <cuda_d3d11_interop.h>

int main() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsResource_t is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsResource_t resource, *resources;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsRegisterFlags is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsRegisterFlagsNone is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsRegisterFlags regFlags = cudaGraphicsRegisterFlagsNone;

  ID3D11Resource* pD3DResource;

  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaGraphicsD3D11RegisterResource is not supported, please try to remigrate with option: --use-experimental-features=bindless_images.
  // CHECK-NEXT: */
  cudaGraphicsD3D11RegisterResource(&resource, pD3DResource, regFlags);

  return 0;
}

#endif
