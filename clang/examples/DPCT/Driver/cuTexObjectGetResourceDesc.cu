void test(CUDA_RESOURCE_DESC *pr) {
  // Start
  CUtexObject t;
  cuTexObjectGetResourceDesc(pr /*CUDA_RESOURCE_DESC **/, t /*CUtexObject*/);
  // End
}
