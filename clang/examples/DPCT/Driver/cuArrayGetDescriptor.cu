// Option: --use-experimental-features=bindless_images

void test(CUDA_ARRAY_DESCRIPTOR *desc, CUarray array) {
  // Start
  cuArrayGetDescriptor(desc /*CUDA_ARRAY_DESCRIPTOR **/, array /*CUarray*/);
  // End
}