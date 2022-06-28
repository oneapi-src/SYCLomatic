// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/AtomicHelper/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/AtomicHelper/api_test2_out/MainSourceFiles.yaml | wc -l > %T/AtomicHelper/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/AtomicHelper/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/AtomicHelper/api_test2_out

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