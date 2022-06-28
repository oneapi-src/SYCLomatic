// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0, v10.1, v10.2
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/AtomicHelper/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/AtomicHelper/api_test1_out/MainSourceFiles.yaml | wc -l > %T/AtomicHelper/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/AtomicHelper/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/AtomicHelper/api_test1_out

// CHECK: 1

#include <cuda/atomic>

// TEST_FEATURE: AtomicHelper_atomic_helper
// TEST_FEATURE: AtomicHelper_atomic_helper_store
// TEST_FEATURE: AtomicHelper_atomic_helper_load


int main(){
  cuda::atomic<int> a;
  cuda::atomic<int> b(0);
  cuda::atomic<int, cuda::thread_scope_block> c(0);
  int ans = c.load();
  a.store(0);
  int ans2 = a.load();
  return 0;
}

