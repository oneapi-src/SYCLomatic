// RUN: dpct --format-range=none -out-root %T/cppnoinline %s --cuda-include-path="%cuda-path/include" -- -xc++
// RUN: FileCheck %s --match-full-lines --input-file %T/cppnoinline/cppnoinline.cpp.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cppnoinline/cppnoinline.cpp.dp.cpp -o %T/cppnoinline/cppnoinline.cpp.dp.o %}

#include <cuda_runtime.h>

// CHECK: #define NOINLINE __attribute__((__noinline__))
#define NOINLINE __attribute__((__noinline__))
// CHECK: NOINLINE void macro_in_macro_with_attr() {}
NOINLINE void macro_in_macro_with_attr() {}
