// RUN: dpct -in-root %S -out-root %T/usertypedef %S/usertypedef.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/usertypedef/usertypedef.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/usertypedef/usertypedef.dp.cpp -o %T/usertypedef/usertypedef.dp.o %}

#ifdef USE_CUDNN
#include <cudnn.h>
#else
typedef void* cudnnHandle_t;
#endif
#include<cuda.h>

int main() {
  // CHECK: cudnnHandle_t handle;
  cudnnHandle_t handle;
}