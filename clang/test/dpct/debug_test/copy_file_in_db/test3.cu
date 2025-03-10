// RUN: mkdir -p %T/common
// RUN: cd %T
// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/common/test1.cpp > %T/common/test1.cpp
// RUN: cat %S/test2.cpp > %T/test2.cpp
// RUN: cat %S/test3.cu > %T/test3.cu
// RUN: cat %S/test.h > %T/test.h
// RUN: dpct --in-root=%T --out-root=%T/out -p %T --format-range=none --cuda-include-path="%cuda-path/include" --enable-codepin
// RUN: FileCheck %S/common/test1.cpp --match-full-lines --input-file %T/out_codepin_cuda/common/test1.cpp
#include "test.h"

__global__ void kernel(){
    float a = float_to_force;
}

int main() {
    kernel<<<1, 1>>>();
    return 0;      
}