// UNSUPPORTED: cuda-8.0, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2
// UNSUPPORTED: v8.0, v11.7, v11.8, v12.0, v12.1, v12.2
// RUN: mkdir %T/fix_internal_error_2
// RUN: dpct --out-root %T/fix_internal_error_2 %s --cuda-include-path="%cuda-path/include" > %T/fix_internal_error_2/output.txt 2>&1 || true
// RUN: grep "dpct internal error" %T/fix_internal_error_2/output.txt | wc -l > %T/fix_internal_error_2/wc_output.txt || true
// RUN: FileCheck %s --match-full-lines --input-file %T/fix_internal_error_2/wc_output.txt
// RUN: rm -rf %T/fix_internal_error_2

// CHECK: 0

<<<<<<< HEAD
=======
// Test description:
// DeviceFunctionDecl::LinkRedecls() may return nullptr. The return value need be checked
// to avoid dereferencing a nullptr.
// Below case can test above scenario.
>>>>>>> upstream_syclomatic/SYCLomatic
namespace test_ns {
#include "cooperative_groups.h"
namespace cg = cooperative_groups;

#define TB(b) cg::thread_block b = cg::this_thread_block();

__global__ void k() {
  cg::thread_block cta = cg::this_thread_block();
  cg::sync(cta);

  int p;
  __threadfence_block();
  __threadfence();
  __threadfence_system();
  __syncthreads_and(p);
  __syncwarp(0xffffffff);
}
} // namespace test_ns
