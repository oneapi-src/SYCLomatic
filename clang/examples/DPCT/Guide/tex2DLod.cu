// Option: --use-experimental-features=bindless_images

template <typename T>
__global__ void test(cudaTextureObject_t t, float f1, float f2, float f3) {
  // Start
  tex2DLod<T>(t /*cudaTextureObject_t*/, f1 /*float*/, f2 /*float*/,
              f3 /*float*/);
  // End
}
