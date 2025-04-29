// Option: --use-experimental-features=bindless_images

void test(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem,
          const cudaExternalMemoryMipmappedArrayDesc *mipmapDesc) {
  // Start
  cudaExternalMemoryGetMappedMipmappedArray(
      mipmap /*cudaMipmappedArray_t **/, extMem /*cudaExternalMemory_t*/,
      mipmapDesc /*const cudaExternalMemoryMipmappedArrayDesc **/);
  // End
}
