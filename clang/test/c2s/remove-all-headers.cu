//RUN: c2s -out-root %T/remove-all-headers %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
//RUN: FileCheck --input-file %T/remove-all-headers/remove-all-headers.dp.cpp --match-full-lines %s
//CHECK:#include <CL/sycl.hpp>
//CHECK:#include <c2s/c2s.hpp>
//CHECK:#include <oneapi/mkl.hpp>
#include <cuda.h>
#include <curand.h>

