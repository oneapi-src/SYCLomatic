// Option: --use-experimental-features=bindless_images

void test() {
  // clang-format off
  // Start
  static texture<float4, 3> tex3;
  cudaMipmappedArray_t pMipMapArr;
  cudaBindTextureToMipmappedArray(tex3 /*const struct texture<T, dim, readMode>*/, pMipMapArr /*cudaMipmappedArray_const_t*/);
  // End
  // clang-format on
}
