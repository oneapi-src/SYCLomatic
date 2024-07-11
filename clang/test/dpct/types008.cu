// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

// RUN: dpct -out-root %T/types008 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types008/types008.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/types008/types008.dp.cpp -o %T/types008/types008.dp.o %}

// CHECK:#include <oneapi/dpl/execution>
// CHECK-NEXT:#include <oneapi/dpl/algorithm>
// CHECK-NEXT:#include <sycl/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
// CHECK-NEXT:#include <dpct/dpl_utils.hpp>
#include <cub/cub.cuh>

// CHECK:void foo(){
// CHECK-NEXT:  oneapi::dpl::reverse_iterator<int *> d_tmp(0);
// CHECK-NEXT:}
void foo(){
  thrust::reverse_iterator<int *> d_tmp(0);
}
