// Option: --use-experimental-features=bindless_images

template <class T>
__global__ void test(T *ptr, float x, float y, int layer,
                     cudaTextureObject_t tex, bool *is_resident) {
  // Start
  tex2DLayered<T>(tex /*cudaTextureObject_t*/, x /*float*/, y /*float*/,
                  layer /*int*/);
  tex2DLayered(ptr /*T **/, tex /*cudaTextureObject_t*/, x /*float*/,
               y /*float*/, layer /*int*/);
  tex2DLayered<T>(tex /*cudaTextureObject_t*/, x /*float*/, y /*float*/,
                  layer /*int*/, is_resident /*bool **/);
  tex2DLayered(ptr /*T **/, tex /*cudaTextureObject_t*/, x /*float*/,
               y /*float*/, layer /*int*/, is_resident /*bool **/);
  // End
}
