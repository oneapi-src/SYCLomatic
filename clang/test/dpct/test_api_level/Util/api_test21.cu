// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test21_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test21_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test21_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test21_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test21_out

// CHECK: 1
// TEST_FEATURE: Util_conj

#include <cuComplex.h>

__device__ void foo1() {
    float2 a;

    auto c = cuConj(a);

}

int main() {
    return 0;
}