// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2
// RUN: dpct --format-range=none --out-root %T/cusparse-10 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cusparse-10/cusparse-10.dp.cpp --match-full-lines %s

#include <cusparse.h>

int main() {
  //CHECK:int alg = 0;
  //CHECK-NEXT:alg = 1;
  cusparseAlgMode_t alg = CUSPARSE_ALG_NAIVE;
  alg = CUSPARSE_ALG_MERGE_PATH;

  return 0;
}
