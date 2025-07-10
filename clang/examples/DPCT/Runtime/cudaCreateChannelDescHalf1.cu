// Option: --use-experimental-features=bindless_images

void test() {
  static texture<float4, 3> tex3;
  cudaMipmappedArray_t pMipMapArr;
  // clang-format off
  // Start
  cudaChannelFormatDesc half1Chn = cudaCreateChannelDescHalf1();
  // End
  // clang-format on
}
