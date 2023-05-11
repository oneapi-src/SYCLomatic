// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test17_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test17_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test17_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test17_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test17_out

// CHECK: 35
// TEST_FEATURE: LapackUtils_syheevx_scratchpad_size

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int64_t n;
  cudaDataType dataTypeA;
  const void *A;
  int64_t lda;
  void *vl;
  void *vu;
  int64_t il;
  int64_t iu;
  int64_t *h_meig;
  cudaDataType dataTypeW;
  const void *W;
  cudaDataType computeType;
  size_t *workspaceInBytesOnDevice;
  size_t *workspaceInBytesOnHost;

  cusolverDnXsyevdx_bufferSize(handle, params, jobz, range, uplo, n, dataTypeA,
                               A, lda, vl, vu, il, iu, h_meig, dataTypeW, W,
                               computeType, workspaceInBytesOnDevice,
                               workspaceInBytesOnHost);
  return 0;
}
