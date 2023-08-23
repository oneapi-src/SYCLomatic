#include <cudnn.h>

void test(cudnnTensorDescriptor_t d, cudnnDataType_t *t, int *n, int *c, int *h,
          int *w, int *ns, int *cs, int *hs, int *ws) {
  // Start
  cudnnGetTensor4dDescriptor(d /*cudnnTensorDescriptor_t*/,
                             t /*cudnnDataType_t **/, n /*int **/, c /*int **/,
                             h /*int **/, w /*int **/, ns /*int **/,
                             cs /*int **/, hs /*int **/, ws /*int **/);
  // End
}
