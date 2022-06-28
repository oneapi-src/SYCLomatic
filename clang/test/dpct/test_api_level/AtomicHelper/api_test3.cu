// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/AtomicHelper/api_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/AtomicHelper/api_test3_out/MainSourceFiles.yaml | wc -l > %T/AtomicHelper/api_test3_out/count.txt
// RUN: FileCheck --input-file %T/AtomicHelper/api_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/AtomicHelper/api_test3_out

#include <cuda/atomic>

// TEST_FEATURE: AtomicHelper_atomic_helper_fetch_add
// TEST_FEATURE: AtomicHelper_atomic_helper_fetch_sub
// TEST_FEATURE: AtomicHelper_atomic_helper_fetch_and
// TEST_FEATURE: AtomicHelper_atomic_helper_fetch_or
// TEST_FEATURE: AtomicHelper_atomic_helper_fetch_xor


int main(){
    cuda::atomic<int> a(0);
    int ans = a.fetch_add(1);
    ans = a.fetch_sub(-1);
    ans = a.fetch_and(2);
    ans = a.fetch_or(1);
    ans = a.fetch_xor(2);
    return 0;
}