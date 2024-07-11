// Option: --use-experimental-features=bindless_images

template <typename T> __global__ void test(cudaTextureObject_t t, int i) {
  // Start
  tex1Dfetch<T>(t /*cudaTextureObject_t*/, i /*int*/);
  // End
}
