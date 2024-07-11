#include <cudnn.h>

void test(cudnnRNNDescriptor_t d, cudnnForwardMode_t m, int32_t sa[],
          cudnnRNNDataDescriptor_t src_d, void *src,
          cudnnRNNDataDescriptor_t dst_d, void *dst,
          cudnnTensorDescriptor_t h_d, void *hx, void *hy,
          cudnnTensorDescriptor_t c_d, void *cx, void *cy,
          size_t weightspace_size, void *weightspace, size_t workspace_size,
          void *workspace, size_t reservespace_size, void *reservespace) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnRNNForward(h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/,
                  m /*cudnnForwardMode_t*/, sa /*int32_t []*/,
                  src_d /*cudnnRNNDataDescriptor_t*/, src /*void **/,
                  dst_d /*cudnnRNNDataDescriptor_t*/, dst /*void **/,
                  h_d /*cudnnTensorDescriptor_t*/, hx /*void **/, hy /*void **/,
                  c_d /*cudnnTensorDescriptor_t*/, cx /*void **/, cy /*void **/,
                  weightspace_size /*size_t*/, weightspace /*void **/,
                  workspace_size /*size_t*/, workspace /*void **/,
                  reservespace_size /*size_t*/, reservespace /*void **/);
  // End
}