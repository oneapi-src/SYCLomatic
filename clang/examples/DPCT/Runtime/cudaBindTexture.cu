// Option: --use-experimental-features=bindless_images

void test(size_t *ps, const textureReference *ptr, const void *pv,
          const cudaChannelFormatDesc *pc, size_t s) {
  // Start
  cudaBindTexture(ps /*size_t **/, ptr /*const textureReference **/,
                  pv /*const void **/, pc /*const cudaChannelFormatDesc **/,
                  s /*size_t*/);
  // End
}
