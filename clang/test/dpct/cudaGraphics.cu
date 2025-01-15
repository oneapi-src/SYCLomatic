// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --use-experimental-features=bindless_images --format-range=none -out-root %T/cudaGraphics %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cudaGraphics/cudaGraphics.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/cudaGraphics/cudaGraphics.dp.cpp -o %T/cudaGraphics/cudaGraphics.dp.o %}

#include <cuda.h>
#ifdef _WIN32
#include <cuda_d3d11_interop.h>
#endif

int main() {
  // CHECK: dpct::experimental::external_mem_wrapper_ptr resource;
  // CHECK-NEXT: dpct::experimental::external_mem_wrapper_ptr *resources;
  // CHECK-NEXT: dpct::experimental::external_mem_wrapper_ptr **resources_ptr;
  cudaGraphicsResource_t resource;
  cudaGraphicsResource_t *resources;
  cudaGraphicsResource_t **resources_ptr;

  // CHECK: dpct::experimental::external_mem_wrapper_ptr resources_arr[10];
  cudaGraphicsResource_t resources_arr[10];

  // CHECK: dpct::experimental::external_mem_wrapper_ptr resource1, *resources1, **resources_ptr1;
  cudaGraphicsResource_t resource1, *resources1, **resources_ptr1;

  cudaMipmappedArray_t mipmappedArray, *mipmappedArray_ptr;
  cudaArray_t array, *array_ptr;

  // CHECK: int regFlags = 0;
  // CHECK-NEXT: int regFlags1 = 0;
  // CHECK-NEXT: int regFlags2 = 0;
  // CHECK-NEXT: int regFlags3 = 0;
  // CHECK-NEXT: int regFlags4 = 0;
  cudaGraphicsRegisterFlags regFlags = cudaGraphicsRegisterFlagsNone;
  cudaGraphicsRegisterFlags regFlags1 = cudaGraphicsRegisterFlagsReadOnly;
  cudaGraphicsRegisterFlags regFlags2 = cudaGraphicsRegisterFlagsWriteDiscard;
  cudaGraphicsRegisterFlags regFlags3 = cudaGraphicsRegisterFlagsSurfaceLoadStore;
  cudaGraphicsRegisterFlags regFlags4 = cudaGraphicsRegisterFlagsTextureGather;

#ifdef _WIN32
  ID3D11Resource *pD3DResource, *pD3DResource1;

  // CHECK-WINDOWS: resource = new dpct::experimental::external_mem_wrapper(pD3DResource, 0);
  cudaGraphicsD3D11RegisterResource(&resource, pD3DResource, cudaGraphicsRegisterFlagsNone);

  // CHECK-WINDOWS: resource1 = new dpct::experimental::external_mem_wrapper(pD3DResource1, regFlags1);
  cudaGraphicsD3D11RegisterResource(&resource1, pD3DResource1, regFlags1);
#endif // _WIN32

  resources_arr[0] = resource;
  resources_arr[1] = resource1;

  void *devPtr;
  size_t size;

  // CHECK: int mapFlags = 0;
  // CHECK-NEXT: int mapFlags1 = 0;
  // CHECK-NEXT: int mapFlags2 = 0;
  cudaGraphicsMapFlags mapFlags = cudaGraphicsMapFlagsNone;
  cudaGraphicsMapFlags mapFlags1 = cudaGraphicsMapFlagsReadOnly;
  cudaGraphicsMapFlags mapFlags2 = cudaGraphicsMapFlagsWriteDiscard;

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaGraphicsResourceSetMapFlags was removed because this functionality is deprecated in DX12 and hence is not supported in SYCL.
  // CHECK-NEXT: */
  cudaGraphicsResourceSetMapFlags(resource, cudaGraphicsMapFlagsNone);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaGraphicsResourceSetMapFlags was removed because this functionality is deprecated in DX12 and hence is not supported in SYCL.
  // CHECK-NEXT: */
  cudaGraphicsResourceSetMapFlags(resource1, mapFlags1);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

#ifdef _WIN32
  // CHECK-WINDOWS: dpct::experimental::map_resources(2, resources_arr, stream);
  cudaGraphicsMapResources(2, resources_arr, stream);

  // CHECK-WINDOWS: dpct::experimental::unmap_resources(2, resources_arr, stream);
  cudaGraphicsUnmapResources(2, resources_arr, stream);

  // CHECK-WINDOWS: dpct::experimental::map_resources(1, &resource);
  cudaGraphicsMapResources(1, &resource);
#endif // _WIN32

  // CHECK: mipmappedArray = resource->get_mapped_mipmapped_array();
  cudaGraphicsResourceGetMappedMipmappedArray(&mipmappedArray, resource);

  // CHECK: *mipmappedArray_ptr = resource->get_mapped_mipmapped_array();
  cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray_ptr, resource);

  unsigned int arrayIndex, mipLevel;
  // CHECK: array = resource->get_sub_resource_mapped_array(arrayIndex, mipLevel);
  cudaGraphicsSubResourceGetMappedArray(&array, resource, arrayIndex, mipLevel);

  // CHECK: *array_ptr = resource->get_sub_resource_mapped_array(arrayIndex, mipLevel);
  cudaGraphicsSubResourceGetMappedArray(array_ptr, resource, arrayIndex, mipLevel);

  // CHECK: resource->get_mapped_pointer(&devPtr, &size);
  cudaGraphicsResourceGetMappedPointer(&devPtr, &size, resource);

#ifdef _WIN32
  // CHECK-WINDOWS: dpct::experimental::unmap_resources(1, &resource);
  cudaGraphicsUnmapResources(1, &resource);
#endif // _WIN32

  // CHECK: delete resource;
  cudaGraphicsUnregisterResource(resource);

  // CHECK: delete resource1;
  cudaGraphicsUnregisterResource(resource1);

  return 0;
}
