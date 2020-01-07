//RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/remove-all-headers.dp.cpp --match-full-lines %s
//CHECK:#include <CL/sycl.hpp>
//CHECK:#include <dpct/dpct.hpp>
//CHECK:#include <mkl_rng_sycl.hpp>
#include <cuda.h>
#include <curand.h>
