__global__
void foo() {
  size_t tix = threadIdx.x;
  size_t tiy = threadIdx.y;
  size_t tiz = threadIdx.z;

  // size_t bix = blockIdx.x;
  // size_t biy = blockIdx.y;
  // size_t biz = blockIdx.z;

  size_t bdx = blockDim.x;
  size_t bdy = blockDim.y;
  size_t bdz = blockDim.z;

  // size_t gdx = gridDim.x;
  // size_t gdy = gridDim.y;
  // size_t gdz = gridDim.z;
}