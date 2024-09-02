// RUN: dpct --format-range=none -out-root %T/code_in_assert %s --extra-arg="-DNDEBUG" --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/code_in_assert/code_in_assert.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST  %T/code_in_assert/code_in_assert.dp.cpp -o %T/code_in_assert/code_in_assert.dp.o %}

#include <cuda_runtime.h>
#include <cassert>

__global__ void kernel()
{
// CHECK: assert(item_ct1.get_local_range(2) * item_ct1.get_group_range(2) <= 10);
    assert(blockDim.x * gridDim.x <= 10);
}

int main() {
  kernel<<<1,1>>>();
  return 0;
}
