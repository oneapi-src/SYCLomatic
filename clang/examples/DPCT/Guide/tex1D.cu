// Option: --use-experimental-features=bindless_images

template <typename T> __global__ void test(cudaTextureObject_t t, float f) {
  // Start
  tex1D<T>(t /*cudaTextureObject_t*/, f /*float*/);
  // End
}
