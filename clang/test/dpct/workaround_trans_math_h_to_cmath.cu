// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/workaround_trans_math_h_to_cmath.dp.cpp --match-full-lines %s
//CHECK:#include <CL/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK://math header
//CHECK: #include <math.h>
//CHECK-NEXT: #include <cmath>
//CHECK-NEXT: #include "math.h"
#include <cuda.h>
//math header
#include <math.h>
#include <cmath>
#include "math.h"
