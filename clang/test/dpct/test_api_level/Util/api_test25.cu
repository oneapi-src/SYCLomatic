// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test25_out %s --cuda-include-path="%cuda-path/include" --use-experimental-features=logical-group -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test25_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test25_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test25_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test25_out

// CHECK: 4
// TEST_FEATURE: Util_logical_group
// TEST_FEATURE: Util_get_sycl_language_version

#include "cuda.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ void foo() {
  cg::thread_block ttb = cg::this_thread_block();
  cg::thread_block_tile<8> tbt = cg::tiled_partition<8>(ttb);
}

void foo2() {
  unsigned int ver;
  CUcontext ctx;
  cuCtxGetApiVersion(ctx, &ver);
}
