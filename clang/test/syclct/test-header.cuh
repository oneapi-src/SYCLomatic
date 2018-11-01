
__global__ void foofunc() {
  // CHECK: size_t tix = item_{{[a-f0-9]+}}.get_local_id(0);
  size_t tix = threadIdx.x;
}
