// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --use-experimental-features=bindless_images --format-range=none -out-root %T/cuGraphics %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cuGraphics/cuGraphics.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/cuGraphics/cuGraphics.dp.cpp -o %T/cuGraphics/cuGraphics.dp.o %}

#include <cuda.h>

int main() {
  // CHECK: dpct::experimental::external_mem_wrapper_ptr resource;
  // CHECK-NEXT: dpct::experimental::external_mem_wrapper_ptr *resources;
  // CHECK-NEXT: dpct::experimental::external_mem_wrapper_ptr **resources_ptr;
  CUgraphicsResource resource;
  CUgraphicsResource *resources;
  CUgraphicsResource **resources_ptr;

  // CHECK: dpct::experimental::external_mem_wrapper_ptr resources_arr[10];
  CUgraphicsResource resources_arr[10];

  // CHECK: dpct::experimental::external_mem_wrapper_ptr resource1, *resources1, **resources_ptr1;
  CUgraphicsResource resource1, *resources1, **resources_ptr1;

  resources_arr[0] = resource;
  resources_arr[1] = resource1;

  CUdeviceptr pDevPtr;
  size_t pSize;

  CUstream stream;
  cuStreamCreate(&stream, 0);

#ifdef _WIN32
  // CHECK-WINDOWS: dpct::experimental::map_resources(2, resources_arr, stream);
  cuGraphicsMapResources(2, resources_arr, stream);

  // CHECK-WINDOWS: dpct::experimental::map_resources(1, &resource, stream);
  cuGraphicsMapResources(1, &resource, stream);
#endif // _WIN32

  // CHECK: resource->get_mapped_pointer((void **)&pDevPtr, &pSize);
  cuGraphicsResourceGetMappedPointer(&pDevPtr, &pSize, resource);

#ifdef _WIN32
  // CHECK-WINDOWS: dpct::experimental::unmap_resources(2, resources_arr, stream);
  cuGraphicsUnmapResources(2, resources_arr, stream);

  // CHECK-WINDOWS: dpct::experimental::unmap_resources(1, &resource, stream);
  cuGraphicsUnmapResources(1, &resource, stream);
#endif // _WIN32

  // CHECK: delete resource;
  cuGraphicsUnregisterResource(resource);

  // CHECK: delete resource1;
  cuGraphicsUnregisterResource(resource1);

  return 0;
}
