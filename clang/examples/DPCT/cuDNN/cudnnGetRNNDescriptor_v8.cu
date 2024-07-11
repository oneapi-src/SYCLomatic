#include <cudnn.h>

void test(cudnnRNNAlgo_t *alg, cudnnRNNMode_t *m, cudnnRNNBiasMode_t *bm,
          cudnnDirectionMode_t *dm, cudnnRNNInputMode_t *im, cudnnDataType_t *t,
          cudnnDataType_t *mp, cudnnMathType_t *mt, int32_t *is, int32_t *hs,
          int32_t *ps, int32_t *l, cudnnDropoutDescriptor_t *dropout,
          uint32_t *f) {
  // Start
  cudnnRNNDescriptor_t d;
  cudnnGetRNNDescriptor_v8(
      d /*cudnnRNNDescriptor_t*/, alg /*cudnnRNNAlgo_t **/,
      m /*cudnnRNNMode_t **/, bm /*cudnnRNNBiasMode_t **/,
      dm /*cudnnDirectionMode_t **/, im /*cudnnRNNInputMode_t **/,
      t /*cudnnDataType_t **/, mp /*cudnnDataType_t **/,
      mt /*cudnnMathType_t **/, is /*int32_t **/, hs /*int32_t **/,
      ps /*int32_t **/, l /*int32_t **/, dropout /*cudnnDropoutDescriptor_t **/,
      f /*uint32_t **/);
  // End
}