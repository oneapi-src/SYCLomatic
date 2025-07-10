// Option: --use-experimental-features=bindless_images
#include <cuda.h>
#include <cuda_runtime.h>

void test() {
  cudaSurfaceObject_t surf;
  // clang-format off
  // Start
  cudaDestroySurfaceObject(surf /*cudaSurfaceObject_t*/);
  // End
  // clang-format on
}

