// Option: --use-experimental-features=bindless_images

#include <cuda_d3d11_interop.h>

void test(cudaGraphicsResource_t *r, ID3D11Resource* pD3Dr, unsigned f) {
  // Start
  cudaGraphicsD3D11RegisterResource(r /*cudaGraphicsResource_t **/,
                                    pD3Dr /*ID3D11Resource **/,
                                    f /*unsigned*/);
  // End
}
