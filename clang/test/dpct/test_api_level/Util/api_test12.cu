// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --use-custom-helper=api --use-experimental-features=nd_range_barrier -out-root %T/Util/api_test12_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test12_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test12_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test12_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test12_out

// CHECK: 2

// TEST_FEATURE: Util_nd_range_barrier

#include "cooperative_groups.h"
namespace cg = cooperative_groups;
using namespace cooperative_groups;

__global__ void kernel() {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
}