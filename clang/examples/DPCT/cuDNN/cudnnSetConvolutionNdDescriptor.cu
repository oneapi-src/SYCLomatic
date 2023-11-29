#include <cudnn.h>

void test(int n, int paddinga[], int stridea[], int dilationa[],
          cudnnConvolutionMode_t m, cudnnDataType_t t) {
  // Start
  cudnnConvolutionDescriptor_t d;
  cudnnSetConvolutionNdDescriptor(
      d /*cudnnConvolutionDescriptor_t*/, n /*int*/, paddinga /*int[]*/,
      stridea /*int[]*/, dilationa /*int[]*/, m /*cudnnConvolutionMode_t*/,
      t /*cudnnDataType_t*/);
  // End
}