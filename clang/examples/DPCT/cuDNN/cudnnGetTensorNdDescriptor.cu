#include <cudnn.h>

void test(int rn, cudnnDataType_t *t, int *n, int da[], int sa[]) {
  // Start
  cudnnTensorDescriptor_t d;
  cudnnGetTensorNdDescriptor(d /*cudnnTensorDescriptor_t*/, rn /*int*/,
                             t /*cudnnDataType_t **/, n /*int **/, da /*int[]*/,
                             sa /*int[]*/);
  // End
}