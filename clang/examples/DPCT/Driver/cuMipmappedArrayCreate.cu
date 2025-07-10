// Option: --use-experimental-features=bindless_images

void test(CUmipmappedArray *array, CUDA_ARRAY3D_DESCRIPTOR *desc,
          unsigned int levels) {
  // Start
  cuMipmappedArrayCreate(array /*CUmipmappedArray **/, desc /*CUDA_ARRAY3D_DESCRIPTOR **/, levels /*unsigned int*/);
  // End
}