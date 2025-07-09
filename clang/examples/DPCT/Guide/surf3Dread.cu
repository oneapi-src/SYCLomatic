// Option: --use-experimental-features=bindless_images
template <typename T>
__global__ void test(float data, cudaSurfaceObject_t surf, int x, int y,
                     int z) {
  // Start
  data /*float*/ = surf3Dread<T>(surf /*cudaSurfaceObject_t*/, x /*int*/,
                                 y /*int*/, z /*int*/);
  // End
}