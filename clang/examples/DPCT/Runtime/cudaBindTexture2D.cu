// Option: --use-experimental-features=bindless_images

void test(size_t *ps, const textureReference *ptr, const void *pv,
          const cudaChannelFormatDesc *pc, size_t s1, size_t s2, size_t s3) {
  // Start
  cudaBindTexture2D(ps /*size_t **/, ptr /*const textureReference **/,
                    pv /*const void **/, pc /*const cudaChannelFormatDesc **/,
                    s1 /*size_t*/, s2 /*size_t*/, s3 /*size_t*/);
  // End
}
