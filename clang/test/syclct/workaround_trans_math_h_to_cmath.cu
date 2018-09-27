// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/workaround_trans_math_h_to_cmath.sycl.cpp --match-full-lines %s
//CHECK:#include <CL/sycl.hpp>
//CHECK-NEXT:#include <syclct/syclct.hpp>
//CHECK://math header
//CHECKT: #include <cmath>
//CHECKT: #include "math.h"
#include <cuda.h>
//math header
#include <math.h>
#include "math.h"
