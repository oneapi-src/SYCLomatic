// RUN: cd %T
// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/test1.cpp > %T/test1.cpp
// RUN: cat %S/test2.cpp > %T/test2.cpp
// RUN: cat %S/test3.cu > %T/test3.cu
// RUN: cat %S/test.h > %T/test.h
// RUN: dpct --in-root=%T --out-root=%T/out -p %T --format-range=none --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %S/test.h --match-full-lines --input-file %T/out/test.h
// RUN: %if build_lit %{icpx -c -fsycl %T/out/test3.dp.cpp -o %T/out/test3.dp.o %}
#include "test.h"

__global__ void kernel(){
    float a = float_to_force;
}

int main() {
    kernel<<<1, 1>>>();
    return 0;      
}