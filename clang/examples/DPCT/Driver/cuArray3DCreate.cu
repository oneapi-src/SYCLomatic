// Option: --use-experimental-features=bindless_images

void test(CUarray *pa, const CUDA_ARRAY3D_DESCRIPTOR *pd) {
  // Start
  cuArray3DCreate(pa /*CUarray **/, pd /*const CUDA_ARRAY3D_DESCRIPTOR **/);
  // End
}
