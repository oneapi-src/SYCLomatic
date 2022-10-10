// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-header3 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: FileCheck --input-file %T/thrust-header3/thrust-header3.dp.cpp --match-full-lines %s

//CHECK: #include <oneapi/dpl/execution>
//CHECK: #include <oneapi/dpl/algorithm>
//CHECK: #include <sycl/sycl.hpp>
//CHECK: #include <dpct/dpct.hpp>
//CHECK: #include <dpct/dpl_utils.hpp>
#include "thrust/functional.h"
#include "thrust/sort.h"
#include "thrust/device_vector.h"


int test() {
    thrust::device_vector<int> a, b, c;
    thrust::sort_by_key(a.begin(), b.end(), c.begin());

    return 0;
}