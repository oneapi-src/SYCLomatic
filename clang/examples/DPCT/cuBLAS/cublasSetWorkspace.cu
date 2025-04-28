#include "cublas_v2.h"

void test(cublasHandle_t handle, void *workspace, size_t size) {
  // Start
  cublasSetWorkspace(handle /*cublasHandle_t*/, workspace /*void **/,
                     size /*size_t*/);
  // End
}
