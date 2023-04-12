// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test28_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test28_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test28_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test28_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test28_out


// CHECK: 2
// TEST_FEATURE: DnnlUtils_get_version

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {

    size_t ver = cudnnGetVersion();
    return 0;

}