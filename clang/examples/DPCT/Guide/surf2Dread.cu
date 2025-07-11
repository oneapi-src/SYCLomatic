// Option: --use-experimental-features=bindless_images
template <typename T>
__global__ void test(float data, cudaSurfaceObject_t surf, int x, int y) {
  // Start
  data /*float*/ =
      surf2Dread<T>(surf /*cudaSurfaceObject_t*/, x /*int*/, y /*int*/);
  // End
}