// RUN: dpct -in-root %S -out-root %T/version %S/version.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/version/version.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>


int main() {
    // CHECK: size_t version = dpct::dnnl::get_version();
    size_t version = cudnnGetVersion();

    return 0;
}