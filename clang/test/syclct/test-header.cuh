
__global__ void foofunc() {
  // CHECK: size_t tix = item.get_local_id(0);
  size_t tix = threadIdx.x;
}
