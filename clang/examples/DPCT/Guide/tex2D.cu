// Option: --use-experimental-features=bindless_images

template <typename T>
__global__ void test(cudaTextureObject_t t, float f1, float f2) {
  // Start
  tex2D<T>(t /*cudaTextureObject_t*/, f1 /*float*/, f2 /*float*/);
  // End
}
