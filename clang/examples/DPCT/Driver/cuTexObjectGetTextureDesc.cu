void test(CUDA_TEXTURE_DESC *pt) {
  // Start
  CUtexObject t;
  cuTexObjectGetTextureDesc(pt /*CUDA_TEXTURE_DESC **/, t /*CUtexObject*/);
  // End
}
