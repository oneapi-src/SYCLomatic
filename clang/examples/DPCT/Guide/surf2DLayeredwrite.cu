// Option: --use-experimental-features=bindless_images
template <typename T>
__global__ void test(float data, cudaSurfaceObject_t surf, int x, int y,
                     int layer) {
  // Start
  surf2DLayeredwrite<T>(data /*float*/, surf /*cudaSurfaceObject_t*/, x /*int*/,
                        y /*int*/, layer /*int*/);
  // End
}