#include <cudnn.h>

void test(cudnnReduceTensorDescriptor_t d, void *i, size_t is, void *w, size_t ws,
          void *alpha, cudnnTensorDescriptor_t src_d, void *src, void *beta,
          cudnnTensorDescriptor_t dst_d, void *dst) {
  // Start
  cudnnHandle_t h;
  cudnnCreate(&h /*cudnnHandle_t **/);
  cudnnReduceTensor(h /*cudnnHandle_t*/, d /*cudnnReduceTensorDescriptor_t*/,
                    i /*void**/, is /*size_t*/, w /*void**/, ws /*size_t*/,
                    alpha /*void **/, src_d /*cudnnTensorDescriptor_t*/,
                    src /*void **/, beta /*void **/,
                    dst_d /*cudnnTensorDescriptor_t*/, dst /*void **/);
  // End
}