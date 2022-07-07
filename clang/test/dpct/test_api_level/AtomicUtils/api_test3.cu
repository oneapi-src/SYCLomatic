// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0, v10.1, v10.2
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/AtomicUtils/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/AtomicUtils/api_test3_out/MainSourceFiles.yaml | wc -l > %T/AtomicUtils/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/AtomicUtils/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/AtomicUtils/api_test3_out

// CHECK: 1

#include <cuda/atomic>

// TEST_FEATURE: AtomicUtils_atomic_utils_fetch_add
// TEST_FEATURE: AtomicUtils_atomic_utils_fetch_sub
// TEST_FEATURE: AtomicUtils_atomic_utils_fetch_and
// TEST_FEATURE: AtomicUtils_atomic_utils_fetch_or
// TEST_FEATURE: AtomicUtils_atomic_utils_fetch_xor


int main(){
  cuda::atomic<int> a(0);
  int ans = a.fetch_add(1);
  ans = a.fetch_sub(-1);
  ans = a.fetch_and(2);
  ans = a.fetch_or(1);
  ans = a.fetch_xor(2);
  return 0;
}