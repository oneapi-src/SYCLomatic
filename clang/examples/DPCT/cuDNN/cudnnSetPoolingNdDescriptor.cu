#include <cudnn.h>

void test(cudnnPoolingMode_t m, cudnnNanPropagation_t p, 
    int nd, int da[], int pa[], int sa[]) {
  // Start
  cudnnPoolingDescriptor_t d;
  cudnnSetPoolingNdDescriptor(d /*cudnnPoolingDescriptor_t*/,
                              m /*cudnnPoolingMode_t*/,
                              p /*cudnnNanPropagation_t*/, nd /*int*/,
                              da /*int[]*/, pa /*int[]*/, sa /*int[]*/);
  // End
}