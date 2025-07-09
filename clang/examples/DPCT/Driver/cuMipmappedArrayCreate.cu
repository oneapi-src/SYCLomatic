// Option: --use-experimental-features=bindless_images

void test(CUmipmappedArray *array, CUDA_ARRAY3D_DESCRIPTOR *desc,
          unsigned int levels) {
  // Start
  cuMipmappedArrayCreate(array, desc, levels);
  // End
}