#include <cudnn.h>

void test(cudnnRNNDescriptor_t d, cudnnWgradMode_t wm, int32_t sa[],
          cudnnRNNDataDescriptor_t dst_d, void *dst, void *diff_dst,
          cudnnRNNDataDescriptor_t src_d, void *src, void *diff_src,
          cudnnTensorDescriptor_t h_d, void *hx, void *diff_hy, void *diff_hx,
          cudnnTensorDescriptor_t c_d, void *cx, void *diff_cy, void *diff_cx,
          size_t weightspace_size, void *weightspace, void *diff_weightspace,
          size_t workspace_size, void *workspace, size_t reservespace_size,
          void *reservespace) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnRNNBackwardData_v8(
      h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/, sa /*int32_t []*/,
      dst_d /*cudnnRNNDataDescriptor_t*/, dst /*void **/, diff_dst /*void **/,
      src_d /*cudnnRNNDataDescriptor_t*/, diff_src /*void **/,
      h_d /*cudnnTensorDescriptor_t*/, hx /*void **/, diff_hy /*void **/,
      diff_hx /*void **/, c_d /*cudnnTensorDescriptor_t*/, cx /*void **/,
      diff_cy /*void **/, diff_cx /*void **/, weightspace_size /*size_t*/,
      weightspace /*void **/, workspace_size /*size_t*/, workspace /*void **/,
      reservespace_size /*size_t*/, reservespace /*void **/);
  cudnnRNNBackwardWeights_v8(
      h /*cudnnHandle_t*/, d /*cudnnRNNDescriptor_t*/, wm /*cudnnWgradMode_t*/,
      sa /*int32_t []*/, src_d /*cudnnRNNDataDescriptor_t*/, src /*void **/,
      h_d /*cudnnTensorDescriptor_t*/, hx /*void **/,
      dst_d /*cudnnRNNDataDescriptor_t*/, dst /*void **/,
      weightspace_size /*size_t*/, diff_weightspace /*void **/,
      workspace_size /*size_t*/, workspace /*void **/,
      reservespace_size /*size_t*/, reservespace /*void **/);
  // End
}