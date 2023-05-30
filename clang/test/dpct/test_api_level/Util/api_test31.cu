// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.2, v10.1, v10.2
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test31_out %s --cuda-include-path="%cuda-path/include" --use-experimental-features=logical-group -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test31_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test31_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test31_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test31_out

// CHECK: 1
// TEST_FEATURE: Util_logical_group_get_group_linear_id

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ void foo() {

  auto block = cg::this_thread_block();
  auto tile32 = cg::tiled_partition<32>(block);
  tile32.meta_group_rank();
}
