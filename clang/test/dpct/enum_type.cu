// RUN: dpct --format-range=none -out-root %T/enum_type %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/enum_type/enum_type.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/enum_type/enum_type.dp.cpp -o %T/enum_type/enum_type.dp.o %}
#include <cuda.h>
int main() {
  // CHECK: sycl::usm::alloc mem_type;
  CUmemorytype mem_type;
  // CHECK: if (static_cast<int>(mem_type) == 0)
  if (mem_type == 0)
    ;
  // CHECK: if (0 == static_cast<int>(mem_type))
  if (0 == mem_type)
    ;
  // CHECK: if (sycl::usm::alloc::host == mem_type)
  if (CU_MEMORYTYPE_HOST == mem_type)
    ;
  // CHECK: if (0 <= static_cast<int>(mem_type))
  if (0 <= mem_type)
    ;
  // CHECK: if (static_cast<int>(mem_type) > 0)
  if (mem_type > 0)
    ;
  return 0;
}
