// Option: --use-experimental-features=bindless_images

template <class T>
__global__ void test(T *ptr, float x, int layer, cudaTextureObject_t tex) {
  // Start
  tex1DLayered<T>(tex /*cudaTextureObject_t*/, x /*float*/, layer /*int*/);
  tex1DLayered(ptr /*T **/, tex /*cudaTextureObject_t*/, x /*float*/,
               layer /*int*/);
  // End
}
