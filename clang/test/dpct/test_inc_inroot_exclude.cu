// RUN: dpct --format-range=none --usm-level=none -out-root %T/test_inc_inroot_exclude %s --cuda-include-path="%cuda-path/include" -extra-arg="-I%S/inc"  --in-root-exclude %S/inc -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/test_inc_inroot_exclude/test_inc_inroot_exclude.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/test_inc_inroot_exclude/test_inc_inroot_exclude.dp.cpp -o %T/test_inc_inroot_exclude/test_inc_inroot_exclude.dp.o %}
#ifndef NO_BUILD_TEST
// CHECK:#include "foo.cuh"
#include "foo.cuh"

// CHECK:#include "no_cuda_syntax.cuh"
#include "no_cuda_syntax.cuh"

// CHECK:#include <no_cuda_syntax.cuh>
#include <no_cuda_syntax.cuh>

void test(){
 foo<<<1,1>>>();
}
#endif