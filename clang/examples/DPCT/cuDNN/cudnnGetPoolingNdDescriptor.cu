#include <cudnn.h>

void test(int rn, cudnnPoolingMode_t *m, cudnnNanPropagation_t *p, int *nd,
          int da[], int pa[], int sa[]) {
  // Start
  cudnnPoolingDescriptor_t d;
  cudnnGetPoolingNdDescriptor(d /*cudnnPoolingDescriptor_t*/, rn /*int*/,
                              m /*cudnnPoolingMode_t**/,
                              p /*cudnnNanPropagation_t**/, nd /*int**/,
                              da /*int[]*/, pa /*int[]*/, sa /*int[]*/);
  // End
}