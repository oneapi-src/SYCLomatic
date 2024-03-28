// Option: --use-experimental-features=bindless_images

void test(const textureReference *ptr, const cudaArray_t a,
          const cudaChannelFormatDesc *pc) {
  // Start
  cudaBindTextureToArray(ptr /*const textureReference **/,
                         a /*const cudaArray_t*/,
                         pc /*const cudaChannelFormatDesc **/);
  // End
}
