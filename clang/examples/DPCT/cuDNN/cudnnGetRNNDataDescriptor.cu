#include <cudnn.h>

void test(cudnnDataType_t *t, cudnnRNNDataLayout_t *l, int *len, int *b, int *v,
          int rlen, int sa[], void *p) {
  // Start
  cudnnRNNDataDescriptor_t d;
  cudnnGetRNNDataDescriptor(
      d /*cudnnRNNDataDescriptor_t*/, t /*cudnnDataType_t **/,
      l /*cudnnRNNDataLayout_t **/, len /*int **/, b /*int **/, v /*int **/,
      rlen /*int*/, sa /*int[]*/, p /*void **/);
  // End
}