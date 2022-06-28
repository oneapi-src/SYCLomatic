// UNSUPPORTED: v7.0, v7.5, v8.0, v9.0, v9.2, v10.0, v10.1, v10.2
// UNSUPPORTED: cuda-7.0, cuda-7.5, cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/AtomicHelper/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/AtomicHelper/api_test2_out/MainSourceFiles.yaml | wc -l > %T/AtomicHelper/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/AtomicHelper/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/AtomicHelper/api_test2_out

// CHECK: 1

#include <cuda/std/atomic>

// TEST_FEATURE: AtomicHelper_atomic_helper_exchange
// TEST_FEATURE: AtomicHelper_atomic_helper_compare_exchange_weak
// TEST_FEATURE: AtomicHelper_atomic_helper_compare_exchange_strong


int main(){
  int tmp =1,tmp1=2;
  cuda::std::atomic<int> a(0);
  int ans = a.exchange(1);
  a.compare_exchange_weak(tmp,2);
  a.compare_exchange_strong(tmp1,3);
  return 0;
}