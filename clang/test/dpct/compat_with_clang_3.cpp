// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1
// RUN: dpct --format-range=none --enable-profiling=0 -out-root %T/compat_with_clang_3 %s --cuda-include-path="%cuda-path/include" --stop-on-parse-err --extra-arg="-xc++"
// RUN: FileCheck %s --match-full-lines --input-file %T/compat_with_clang_3/compat_with_clang_3.cpp.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/compat_with_clang_3/compat_with_clang_3.cpp.dp.cpp -o %T/compat_with_clang_3/compat_with_clang_3.o.dp.o %}

#ifdef __CUDACC_VER_MAJOR__
#define WMMA_ENABLED
#endif

// CHECK: #ifdef WMMA_ENABLED
// CHECK-NEXT: #include <mma.h>
// CHECK-NEXT: using Type = nvcuda::wmma::col_major;
// CHECK-NEXT: #endif
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#ifdef WMMA_ENABLED
#include <mma.h>
using Type = nvcuda::wmma::col_major;
#endif
#include "cuda_runtime.h"
