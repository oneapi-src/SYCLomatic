// Option: --use-experimental-features=bindless_images
template <typename T>
__global__ void test(float data, cudaSurfaceObject_t surf, int x, int y,
                     int z) {
  // Start
  surf3Dwrite<T>(data /*float*/, surf /*cudaSurfaceObject_t*/, x /*int*/,
                 y /*int*/, z /*int*/);
  // End
}