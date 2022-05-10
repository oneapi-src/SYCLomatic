// RUN: dpct -out-root %T/operator_overload %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/operator_overload/h.h --match-full-lines %S/h.h

#include <iostream>
#include "cublas_v2.h"
#include "h.h"

int main() {
  cuComplex a;
  std::cout << a;
  return 0;
}

