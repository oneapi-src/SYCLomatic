// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGraphicsD3D11RegisterResource | FileCheck %s -check-prefix=CUDA_GRAPHICS_RESOURCE_D33D11_REGISTER_RESOURCE
// CUDA_GRAPHICS_RESOURCE_D33D11_REGISTER_RESOURCE: CUDA API:
// CUDA_GRAPHICS_RESOURCE_D33D11_REGISTER_RESOURCE-NEXT:    cudaGraphicsD3D11RegisterResource(r /*cudaGraphicsResource_t **/,
// CUDA_GRAPHICS_RESOURCE_D33D11_REGISTER_RESOURCE-NEXT:                                      pD3Dr /*ID3D11Resource **/,
// CUDA_GRAPHICS_RESOURCE_D33D11_REGISTER_RESOURCE-NEXT:                                      f /*unsigned*/);
// CUDA_GRAPHICS_RESOURCE_D33D11_REGISTER_RESOURCE-NEXT: On Windows, is migrated to (with the option --use-experimental-features=bindless_images):
// CUDA_GRAPHICS_RESOURCE_D33D11_REGISTER_RESOURCE-NEXT:    r = new dpct::experimental::external_mem_wrapper(pD3Dr, f);
