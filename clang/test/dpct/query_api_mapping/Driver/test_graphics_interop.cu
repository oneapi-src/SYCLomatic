// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuGraphicsMapResources | FileCheck %s -check-prefix=CU_GRAPHICS_MAP_RESOURCES
// CU_GRAPHICS_MAP_RESOURCES: CUDA API:
// CU_GRAPHICS_MAP_RESOURCES-NEXT:    cuGraphicsMapResources(c /*int*/,
// CU_GRAPHICS_MAP_RESOURCES-NEXT:                           r /*CUgraphicsResource **/,
// CU_GRAPHICS_MAP_RESOURCES-NEXT:                           s /*CUstream*/);
// CU_GRAPHICS_MAP_RESOURCES-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CU_GRAPHICS_MAP_RESOURCES-NEXT:    dpct::experimental::map_resources(c, r, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuGraphicsResourceGetMappedPointer | FileCheck %s -check-prefix=CU_GRAPHICS_RESOURCE_GET_MAPPED_POINTER
// CU_GRAPHICS_RESOURCE_GET_MAPPED_POINTER: CUDA API:
// CU_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT:    cuGraphicsResourceGetMappedPointer(&ptr /*CUdeviceptr **/,
// CU_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT:                                       s /*size_t **/,
// CU_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT:                                       r /*CUgraphicsResource*/);
// CU_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CU_GRAPHICS_RESOURCE_GET_MAPPED_POINTER-NEXT:    r->get_mapped_pointer((void **)&ptr, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuGraphicsUnmapResources | FileCheck %s -check-prefix=CU_GRAPHICS_UNMAP_RESOURCES
// CU_GRAPHICS_UNMAP_RESOURCES: CUDA API:
// CU_GRAPHICS_UNMAP_RESOURCES-NEXT:    cuGraphicsUnmapResources(c /*int*/,
// CU_GRAPHICS_UNMAP_RESOURCES-NEXT:                             r /*CUgraphicsResource **/,
// CU_GRAPHICS_UNMAP_RESOURCES-NEXT:                             s /*CUstream*/);
// CU_GRAPHICS_UNMAP_RESOURCES-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CU_GRAPHICS_UNMAP_RESOURCES-NEXT:    dpct::experimental::unmap_resources(c, r, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuGraphicsUnregisterResource | FileCheck %s -check-prefix=CU_GRAPHICS_UNREGISTER_RESOURCE
// CU_GRAPHICS_UNREGISTER_RESOURCE: CUDA API:
// CU_GRAPHICS_UNREGISTER_RESOURCE-NEXT:    cuGraphicsUnregisterResource(r /*CUgraphicsResource*/);
// CU_GRAPHICS_UNREGISTER_RESOURCE-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CU_GRAPHICS_UNREGISTER_RESOURCE-NEXT:    delete r;
