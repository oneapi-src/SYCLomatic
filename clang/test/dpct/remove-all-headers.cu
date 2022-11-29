//RUN: dpct -out-root %T/remove-all-headers %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/remove-all-headers/remove-all-headers.dp.cpp --match-full-lines %s
//CHECK:#include <sycl/sycl.hpp>
//CHECK:#include <dpct/dpct.hpp>
//CHECK:#include <dpct/rng_utils.hpp>
#include <cuda.h>
#include <curand.h>

