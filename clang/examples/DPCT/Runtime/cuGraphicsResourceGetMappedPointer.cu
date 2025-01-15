// Option: --use-experimental-features=bindless_images

void test(CUdeviceptr ptr, size_t *s, CUgraphicsResource r) {
  // Start
  cuGraphicsResourceGetMappedPointer(&ptr /*CUdeviceptr **/,
                                      s /*size_t **/,
                                      r /*CUgraphicsResource*/);
  // End
}
