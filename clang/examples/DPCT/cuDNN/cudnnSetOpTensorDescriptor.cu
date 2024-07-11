#include <cudnn.h>

void test(cudnnOpTensorOp_t op, cudnnDataType_t dt, cudnnNanPropagation_t p) {
  // Start
  cudnnOpTensorDescriptor_t d;
  cudnnSetOpTensorDescriptor(d /*cudnnOpTensorDescriptor_t*/,
                             op /*cudnnOpTensorOp_t*/, dt /*cudnnDataType_t*/,
                             p /*cudnnNanPropagation_t*/);
  // End
}