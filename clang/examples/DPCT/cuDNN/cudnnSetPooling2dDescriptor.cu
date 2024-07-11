#include <cudnn.h>

void test(cudnnPoolingMode_t m, cudnnNanPropagation_t p, 
    int h, int w, int vp, int hp, int vs, int hs) {
  // Start
  cudnnPoolingDescriptor_t d;
  cudnnSetPooling2dDescriptor(d /*cudnnPoolingDescriptor_t*/,
                              m /*cudnnPoolingMode_t*/,
                              p /*cudnnNanPropagation_t*/, h /*int*/, w /*int*/,
                              vp /*int*/, hp /*int*/, vs /*int*/, hs /*int*/);
  // End
}