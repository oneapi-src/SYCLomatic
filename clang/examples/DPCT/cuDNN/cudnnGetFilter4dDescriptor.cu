#include <cudnn.h>

void test(cudnnDataType_t *t, cudnnTensorFormat_t *f, int *k, int *c, int *h,
          int *w) {
  // Start
  cudnnFilterDescriptor_t d;
  cudnnGetFilter4dDescriptor(d /*cudnnFilterDescriptor_t*/,
                             t /*cudnnDataType_t **/,
                             f /*cudnnTensorFormat_t **/, k /*int **/,
                             c /*int **/, h /*int **/, w /*int **/);
  // End
}