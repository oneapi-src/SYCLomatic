// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Util/api_test18_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Util/api_test18_out/MainSourceFiles.yaml | wc -l > %T/Util/api_test18_out/count.txt
// RUN: FileCheck --input-file %T/Util/api_test18_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Util/api_test18_out

// CHECK: 1
// TEST_FEATURE: Util_cdiv

#include <cuComplex.h>

__device__ void foo1() {
    float2 a, b;

    auto c = cuCdiv(a, b);

}

int main() {
    return 0;
}