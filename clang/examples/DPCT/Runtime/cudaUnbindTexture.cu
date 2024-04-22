// Option: --use-experimental-features=bindless_images

void test(const textureReference *ptr) {
  // Start
  cudaUnbindTexture(ptr /*const textureReference **/);
  // End
}
