// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-header2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: FileCheck --input-file %T/thrust-header2/thrust-header2.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/thrust-header2/thrust-header2.h --match-full-lines %S/thrust-header2.h

//CHECK:#include <oneapi/dpl/execution>
//CHECK-NEXT:#include <oneapi/dpl/algorithm>
//CHECK-NEXT:#include <CL/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include "thrust-header2.h"
//CHECK-NEXT:#include <cstdio>
//CHECK-NEXT:#include <dpct/dpl_utils.hpp>
//CHECK-EMPTY:
//CHECK-NEXT:#include <algorithm>
#include "thrust-header2.h"
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <algorithm>

void foo() {
}

