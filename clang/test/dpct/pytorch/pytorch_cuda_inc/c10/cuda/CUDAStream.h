#include <cuda_runtime.h>

namespace c10 {
namespace cuda {
class CUDAStream {
public:
  CUDAStream() {}
  cudaStream_t stream() {}
};

CUDAStream getCurrentCUDAStream() { return CUDAStream(); }
CUDAStream getCurrentCUDAStream(int dev_ind) { return CUDAStream(); }
} // namespace cuda
} // namespace c10
