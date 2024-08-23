// Option: --use-experimental-features=bindless_images

void test(CUarray *pa, const CUDA_ARRAY_DESCRIPTOR *pd) {
  // Start
  cuArrayCreate(pa /*CUarray **/, pd /*const CUDA_ARRAY_DESCRIPTOR **/);
  // End
}
