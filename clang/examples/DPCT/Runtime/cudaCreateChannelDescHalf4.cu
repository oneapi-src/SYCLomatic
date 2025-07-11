// Option: --use-experimental-features=bindless_images

void test() {
  static texture<float4, 3> tex3;
  cudaMipmappedArray_t pMipMapArr;
  // clang-format off
  // Start
  cudaChannelFormatDesc half4Chn = cudaCreateChannelDescHalf4();
  // End
  // clang-format on
}
