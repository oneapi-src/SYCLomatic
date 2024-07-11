#include <cudnn.h>

void test(cudnnOpTensorDescriptor_t d, void *alpha1,
          cudnnTensorDescriptor_t src1_d, void *src1, void *alpha2,
          cudnnTensorDescriptor_t src2_d, void *src2, void *beta,
          cudnnTensorDescriptor_t dst_d, void *dst) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnOpTensor(
      h /*cudnnHandle_t*/, d /*cudnnOpTensorDescriptor_t*/, alpha1 /*void **/,
      src1_d /*cudnnTensorDescriptor_t*/, src1 /*void **/, alpha2 /*void **/,
      src2_d /*cudnnTensorDescriptor_t*/, src2 /*void **/, beta /*void **/,
      dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
  // End
}