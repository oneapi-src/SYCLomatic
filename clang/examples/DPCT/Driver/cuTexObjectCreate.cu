void test(CUtexObject *pt, const CUDA_RESOURCE_DESC *pr,
          const CUDA_TEXTURE_DESC *ptd, const CUDA_RESOURCE_VIEW_DESC *prv) {
  // Start
  cuTexObjectCreate(pt /*CUtexObject **/, pr /*const CUDA_RESOURCE_DESC **/,
                    ptd /*const CUDA_TEXTURE_DESC **/,
                    prv /*const CUDA_RESOURCE_VIEW_DESC **/);
  // End
}
