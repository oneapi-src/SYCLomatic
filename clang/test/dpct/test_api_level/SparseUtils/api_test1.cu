// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/SparseUtils/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/SparseUtils/api_test1_out/MainSourceFiles.yaml | wc -l > %T/SparseUtils/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/SparseUtils/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/SparseUtils/api_test1_out

// CHECK: 2
// TEST_FEATURE: SparseUtils_sparse_matrix_info

#include "cusparse.h"

int main() {
  cusparseMatDescr_t a;
  return 0;
}
