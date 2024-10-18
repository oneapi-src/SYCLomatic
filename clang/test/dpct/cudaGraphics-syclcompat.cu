// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --use-experimental-features=bindless_images --use-syclcompat --format-range=none -out-root %T/cudaGraphics-syclcompat %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaGraphics-syclcompat/cudaGraphics-syclcompat.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/cudaGraphics-syclcompat/cudaGraphics-syclcompat.dp.cpp -o %T/cudaGraphics-syclcompat/cudaGraphics-syclcompat.dp.o %}

#include <cuda.h>
#ifdef _WIN32
#include <cuda_d3d11_interop.h>
#endif

#ifndef NO_BUILD_TEST
int main() {
  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaGraphicsResource_t" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaGraphicsResource_t resource;

  cudaMipmappedArray_t mipmappedArray;
  cudaArray_t array, *array_ptr;

#ifdef _WIN32
  ID3D11Resource *pD3DResource;

  // CHECK-WINDOWS: DPCT1131:{{[0-9]+}}: The migration of "cudaGraphicsD3D11RegisterResource" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaGraphicsD3D11RegisterResource(&resource, pD3DResource, cudaGraphicsRegisterFlagsNone);
#endif

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaGraphicsResourceSetMapFlags" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaGraphicsResourceSetMapFlags(resource, cudaGraphicsMapFlagsNone);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaGraphicsUnmapResources" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaGraphicsUnmapResources(2, &resource, stream);

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaGraphicsMapResources" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaGraphicsMapResources(1, &resource);

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaGraphicsResourceGetMappedMipmappedArray" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaGraphicsResourceGetMappedMipmappedArray(&mipmappedArray, resource);

  unsigned int arrayIndex, mipLevel;
  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaGraphicsSubResourceGetMappedArray" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaGraphicsSubResourceGetMappedArray(&array, resource, arrayIndex, mipLevel);

  // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cudaGraphicsUnregisterResource" is not currently supported with SYCLcompat. Please adjust the code manually.
  cudaGraphicsUnregisterResource(resource);

  return 0;
}
#endif
