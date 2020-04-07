// RUN: echo "[" > %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/main.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/main.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/bar/util.gpu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/bar/util.gpu\"" >> %T/compile_commands.json
// RUN: echo "    }" >> %T/compile_commands.json
// RUN: echo "]\n" >> %T/compile_commands.json

// RUN: dpct --format-range=none --cuda-include-path="%cuda-path/include" -in-root=%S -out-root=%T -p=%T %s %S/bar/util.gpu  --sycl-named-lambda -extra-arg="-I%S/bar" 
// RUN: FileCheck %s --match-full-lines --input-file %T/main.dp.cpp
// RUN: FileCheck %S/bar/util.gpu --match-full-lines --input-file %T/bar/util.gpu.dp.cpp
// RUN: FileCheck %S/bar/util.gpuhead --match-full-lines --input-file %T/bar/util.gpuhead

// RUN: dpct --format-range=none --cuda-include-path="%cuda-path/include" -in-root=%S -out-root=%T  %S/main.gpu   --sycl-named-lambda 
// RUN: FileCheck %S/main.gpu --match-full-lines --input-file %T/main.gpu.dp.cpp

#include <stdio.h>
#include <cuda_runtime.h>

// CHECK:#include "util.gpuhead"
#include "util.gpuhead"

int main(){
 int *a, *b;
 cudaMalloc((void **)&a, 4);
 cudaMalloc((void **)&b, 4);
 kernel_util<<<1,1>>>(a,b);
}
