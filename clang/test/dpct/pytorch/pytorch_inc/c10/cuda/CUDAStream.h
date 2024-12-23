#include <cuda_runtime.h>
#include "CUDAFunctions.h"

namespace c10 {
namespace cuda {
class CUDAStream {
public:
  CUDAStream() {}
  cudaStream_t stream() { return 0; }

  operator cudaStream_t() const {
    return stream();
  }
  cudaStream_t stream() const;
};

CUDAStream getCurrentCUDAStream(DeviceIndex device_index = -1) {
  return CUDAStream();
}

} // namespace cuda
} // namespace c10
