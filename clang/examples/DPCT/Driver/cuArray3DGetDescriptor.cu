// Option: --use-experimental-features=bindless_images

void test(CUDA_ARRAY3D_DESCRIPTOR *desc, CUarray array) {
  // Start
  cuArray3DGetDescriptor(desc /*CUDA_ARRAY3D_DESCRIPTOR **/, array /*CUarray*/);
  // End
}