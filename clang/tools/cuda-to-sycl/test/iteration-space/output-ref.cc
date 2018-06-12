__global__
void foo() {
  size_t tix = item.get_local(0);
  size_t tiy = item.get_local(1);
  size_t tiz = item.get_local(2);

  // size_t bix = blockIdx.x;
  // size_t biy = blockIdx.y;
  // size_t biz = blockIdx.z;

  size_t bdx = item.get_local_range().get(0);
  size_t bdy = item.get_local_range().get(1);
  size_t bdz = item.get_local_range().get(2);

  // size_t gdx = gridDim.x;
  // size_t gdy = gridDim.y;
  // size_t gdz = gridDim.z;
}