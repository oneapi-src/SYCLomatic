#include <cudnn.h>

void test(int rn, int *n, int pada[], int stridea[], int dilationa[],
          cudnnConvolutionMode_t *m, cudnnDataType_t *t) {
  // Start
  cudnnConvolutionDescriptor_t d;
  cudnnGetConvolutionNdDescriptor(
      d /*cudnnConvolutionDescriptor_t*/, rn /*int*/, n /*int**/,
      pada /*int[]*/, stridea /*int[]*/, dilationa /*int[]*/,
      m /*cudnnConvolutionMode_t**/, t /*cudnnDataType_t**/);
  // End
}