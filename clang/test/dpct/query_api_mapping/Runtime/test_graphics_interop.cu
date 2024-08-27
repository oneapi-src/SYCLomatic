// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphicsResourceSetMapFlags | FileCheck %s -check-prefix=CUDA_GRAPHICS_RESOURCE_SET_MAP_FLAGS
// CUDA_GRAPHICS_RESOURCE_SET_MAP_FLAGS: CUDA API:
// CUDA_GRAPHICS_RESOURCE_SET_MAP_FLAGS-NEXT:   cudaGraphicsResourceSetMapFlags(r /*cudaGraphicsResource_t*/,
// CUDA_GRAPHICS_RESOURCE_SET_MAP_FLAGS-NEXT:                                   f /*unsigned*/);
// CUDA_GRAPHICS_RESOURCE_SET_MAP_FLAGS-NEXT: The API is Removed.
// CUDA_GRAPHICS_RESOURCE_SET_MAP_FLAGS-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphicsMapResources | FileCheck %s -check-prefix=CUDA_GRAPHICS_MAP_RESOURCES
// CUDA_GRAPHICS_MAP_RESOURCES: CUDA API:
// CUDA_GRAPHICS_MAP_RESOURCES-NEXT:    cudaGraphicsMapResources(c /*int*/,
// CUDA_GRAPHICS_MAP_RESOURCES-NEXT:                             r /*cudaGraphicsResource_t **/);
// CUDA_GRAPHICS_MAP_RESOURCES-NEXT:    cudaGraphicsMapResources(c /*int*/,
// CUDA_GRAPHICS_MAP_RESOURCES-NEXT:                             r /*cudaGraphicsResource_t **/,
// CUDA_GRAPHICS_MAP_RESOURCES-NEXT:                             s /*cudaStream_t*/);
// CUDA_GRAPHICS_MAP_RESOURCES-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_GRAPHICS_MAP_RESOURCES-NEXT:    dpct::experimental::map_resources(c, r);
// CUDA_GRAPHICS_MAP_RESOURCES-NEXT:    dpct::experimental::map_resources(c, r, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphicsResourceGetMappedPointer | FileCheck %s -check-prefix=CUDA_GRAPHICS_RESOURCE_GET_MAPPED_POINTER
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_POINTER: CUDA API:
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT:    cudaGraphicsResourceGetMappedPointer(&ptr /*void ***/,
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT:                                         s /*size_t **/,
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT:                                         r /*cudaGraphicsResource_t*/);
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT:    r->get_mapped_pointer(&ptr, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphicsResourceGetMappedMipmappedArray | FileCheck %s -check-prefix=CUDA_GRAPHICS_RESOURCE_GET_MAPPED_MIPMAPPED_ARRAY
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_MIPMAPPED_ARRAY: CUDA API:
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_MIPMAPPED_ARRAY-NEXT:    cudaGraphicsResourceGetMappedMipmappedArray(&m /*cudaMipmappedArray_t **/,
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_MIPMAPPED_ARRAY-NEXT:                                                r /*cudaGraphicsResource_t*/);
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_MIPMAPPED_ARRAY-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_GRAPHICS_RESOURCE_GET_MAPPED_MIPMAPPED_ARRAY-NEXT:    m = r->get_mapped_mipmapped_array();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphicsSubResourceGetMappedArray | FileCheck %s -check-prefix=CUDA_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY
// CUDA_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY: CUDA API:
// CUDA_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY-NEXT:    cudaGraphicsSubResourceGetMappedArray(&a /*cudaArray_t **/,
// CUDA_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY-NEXT:                                          r /*cudaGraphicsResource_t*/,
// CUDA_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY-NEXT:                                          i /*unsigned*/,
// CUDA_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY-NEXT:                                          l /*unsigned*/);
// CUDA_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY-NEXT:    a = r->get_sub_resource_mapped_array(i, l);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphicsUnmapResources | FileCheck %s -check-prefix=CUDA_GRAPHICS_UNMAP_RESOURCES
// CUDA_GRAPHICS_UNMAP_RESOURCES: CUDA API:
// CUDA_GRAPHICS_UNMAP_RESOURCES-NEXT:    cudaGraphicsUnmapResources(c /*int*/,
// CUDA_GRAPHICS_UNMAP_RESOURCES-NEXT:                             r /*cudaGraphicsResource_t **/);
// CUDA_GRAPHICS_UNMAP_RESOURCES-NEXT:    cudaGraphicsUnmapResources(c /*int*/,
// CUDA_GRAPHICS_UNMAP_RESOURCES-NEXT:                             r /*cudaGraphicsResource_t **/,
// CUDA_GRAPHICS_UNMAP_RESOURCES-NEXT:                             s /*cudaStream_t*/);
// CUDA_GRAPHICS_UNMAP_RESOURCES-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_GRAPHICS_UNMAP_RESOURCES-NEXT:    dpct::experimental::unmap_resources(c, r);
// CUDA_GRAPHICS_UNMAP_RESOURCES-NEXT:    dpct::experimental::unmap_resources(c, r, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphicsUnregisterResource | FileCheck %s -check-prefix=CUDA_GRAPHICS_UNREGISTER_RESOURCE
// CUDA_GRAPHICS_UNREGISTER_RESOURCE: CUDA API:
// CUDA_GRAPHICS_UNREGISTER_RESOURCE-NEXT:    cudaGraphicsUnregisterResource(r /*cudaGraphicsResource_t*/);
// CUDA_GRAPHICS_UNREGISTER_RESOURCE-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_GRAPHICS_UNREGISTER_RESOURCE-NEXT:    delete r;
