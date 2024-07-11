#include <cudnn.h>

void test(int rn, cudnnDataType_t *t, cudnnTensorFormat_t *f, int *n,
          int da[]) {
  // Start
  cudnnFilterDescriptor_t d;
  cudnnGetFilterNdDescriptor(d /*cudnnFilterDescriptor_t*/, rn /*int*/,
                             t /*cudnnDataType_t*/, f /*cudnnTensorFormat_t*/,
                             n /*int **/, da /*int[]*/);
  // End
}