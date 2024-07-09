// Option: --use-experimental-features=bindless_images

template <typename T>
__global__ void test(cudaTextureObject_t t, float f1, float f2, float f3,
                     float f4) {
  // Start
  tex3DLod<T>(t /*cudaTextureObject_t*/, f1 /*float*/, f2 /*float*/,
              f3 /*float*/, f4 /*float*/);
  // End
}
