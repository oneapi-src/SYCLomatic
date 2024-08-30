// RUN: dpct -out-root %T/kernel-function-typecast %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/kernel-function-typecast/kernel-function-typecast.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/kernel-function-typecast/kernel-function-typecast.dp.cpp -o %T/kernel-function-typecast/kernel-function-typecast.dp.o %}

#include <cstdint>
#include <cuda.h>

typedef uint64_t u64;
u64 foo(CUfunction cuFunc, CUmodule cuMod) {
  cuModuleGetFunction(&cuFunc, cuMod, "kfoo");
  u64 function = (u64)cuFunc;
  return function;
}
