// UNSUPPORTED: -linux-
// RUN: echo "[" > %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile %T/main.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%T/main.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile %T/bar/util_bar.cc\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%T/bar/util_bar.cc\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile %T/bar/util.gpu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%T/bar/util.gpu\"" >> %T/compile_commands.json
// RUN: echo "    }" >> %T/compile_commands.json
// RUN: echo "]\n" >> %T/compile_commands.json

// RUN: cat %S/main.cu > %T/main.cu
// RUN: cat %S/main.gpu > %T/main.gpu
// RUN: cp -r %S/bar %T

// RUN: sed -i  '3,5s/\\/\//g'  %T/compile_commands.json
// RUN: sed -i  '8,10s/\\/\//g'  %T/compile_commands.json 
// RUN: sed -i  '13,15s/\\/\//g'  %T/compile_commands.json 

// RUN: dpct --format-range=none --cuda-include-path="%cuda-path/include" -in-root=%T -out-root=%T/out -p=%T %T/main.cu %T/bar/util.gpu %T/bar/util_bar.cc --sycl-named-lambda -extra-arg="-I%T/bar" -extra-arg="--std=c++14"
// RUN: FileCheck %s --match-full-lines --input-file %T/out/main.dp.cpp
// RUN: FileCheck %S/bar/util.gpu --match-full-lines --input-file %T/out/bar/util.gpu.dp.cpp
// RUN: FileCheck %S/bar/util.gpuhead --match-full-lines --input-file %T/out/bar/util.gpuhead.dp.cpp
// RUN: FileCheck %S/bar/util_bar.hh --match-full-lines --input-file %T/out/bar/util_bar.hh
// RUN: FileCheck %S/bar/macro_def.hh --match-full-lines --input-file %T/out/bar/macro_def.hh
// RUN: FileCheck %S/bar/util_bar.cc --match-full-lines --input-file %T/out/bar/util_bar.cc.dp.cpp


#include <stdio.h>
#include <cuda_runtime.h>

// CHECK:#include "util.gpuhead"
#include "util.gpuhead"

// CHECK:#include "util_bar.hh"
#include "util_bar.hh"

// CHECK:void FooKernel() {
__global__ void FooKernel() {
   foo_util();
   util_bar();
}

int main(){
 int *a, *b;
 cudaMalloc((void **)&a, 4);
 cudaMalloc((void **)&b, 4);
 kernel_util<<<1,1>>>(a,b);
 FooKernel<<<1,1>>>();
}
