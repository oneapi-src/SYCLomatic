// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/LapackUtils/api_test18_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/LapackUtils/api_test18_out/MainSourceFiles.yaml | wc -l > %T/LapackUtils/api_test18_out/count.txt
// RUN: FileCheck --input-file %T/LapackUtils/api_test18_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/LapackUtils/api_test18_out

// CHECK: 40
// TEST_FEATURE: LapackUtils_syheevx

#include "cusolverDn.h"

int main() {
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  cusolverEigMode_t jobz;
  cusolverEigRange_t range;
  cublasFillMode_t uplo;
  int64_t n;
  cudaDataType dataTypeA;
  void *A;
  int64_t lda;
  void *vl;
  void *vu;
  int64_t il;
  int64_t iu;
  int64_t *meig64;
  cudaDataType dataTypeW;
  void *W;
  cudaDataType computeType;
  void *bufferOnDevice;
  size_t workspaceInBytesOnDevice;
  void *bufferOnHost;
  size_t workspaceInBytesOnHost;
  int *info;

  cusolverDnXsyevdx(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl,
                    vu, il, iu, meig64, dataTypeW, W, computeType,
                    bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                    workspaceInBytesOnHost, info);
  return 0;
}
