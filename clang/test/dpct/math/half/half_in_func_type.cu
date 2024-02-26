// RUN: dpct --format-range=none -out-root %T/math/half/ %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/half/half_in_func_type.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/math/half/half_in_func_type.dp.cpp -o %T/math/half/half_in_func_type.dp.o %}

#include "cuda_fp16.h"
#include <functional>

template <typename T> void f(const std::function<T(T)> f) {}

// CHECK: template void f<sycl::half>(const std::function<sycl::half(sycl::half)> f);
template void f<__half>(const std::function<__half(__half)> f);
