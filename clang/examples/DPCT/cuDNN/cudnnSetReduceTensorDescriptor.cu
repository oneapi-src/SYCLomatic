#include <cudnn.h>

void test(cudnnReduceTensorOp_t o, cudnnDataType_t dt, cudnnNanPropagation_t p,
          cudnnReduceTensorIndices_t i, cudnnIndicesType_t it) {
  // Start
  cudnnReduceTensorDescriptor_t d;
  cudnnSetReduceTensorDescriptor(
      d /*cudnnReduceTensorDescriptor_t */, o /*cudnnPoolingMode_t*/,
      dt /*cudnnDataType_t*/, p /*cudnnNanPropagation_t*/,
      i /*cudnnReduceTensorIndices_t*/, it /*cudnnIndicesType_t*/);
  // End
}