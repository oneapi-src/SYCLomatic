void test(cudaTextureObject_t *pto, const cudaResourceDesc *prd,
          const cudaTextureDesc *ptd, const cudaResourceViewDesc *prvd) {
  // Start
  cudaCreateTextureObject(
      pto /*cudaTextureObject_t **/, prd /*const cudaResourceDesc **/,
      ptd /*const cudaTextureDesc **/, prvd /*const cudaResourceViewDesc **/);
  // End
}
