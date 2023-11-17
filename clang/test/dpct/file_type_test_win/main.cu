// UNSUPPORTED: system-linux
// RUN: echo "[" > %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile %S/main.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/main.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile %S/bar/util_bar.cc\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/bar/util_bar.cc\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile %S/bar/util.gpu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/bar/util.gpu\"" >> %T/compile_commands.json
// RUN: echo "    }" >> %T/compile_commands.json
// RUN: echo "]\n" >> %T/compile_commands.json

// RUN: sed -i  '3,5s/\\/\//g'  %T/compile_commands.json
// RUN: sed -i  '8,10s/\\/\//g'  %T/compile_commands.json 
// RUN: sed -i  '13,15s/\\/\//g'  %T/compile_commands.json 

// RUN: dpct --format-range=none --cuda-include-path="%cuda-path/include" -in-root=%S -out-root=%T -p=%T %s %S/bar/util.gpu %S/bar/util_bar.cc --sycl-named-lambda -extra-arg="-I%S/bar" -extra-arg="--std=c++14"
// RUN: echo "=====main.dp.cpp begin====="
// RUN: cat %T/main.dp.cpp
// RUN: echo "=====main.dp.cpp end====="
// RUN: FileCheck %s --match-full-lines --input-file %T/main.dp.cpp
// RUN: echo "=====util.gpu.dp.cpp begin====="
// RUN: cat %T/bar/util.gpu.dp.cpp
// RUN: echo "=====util.gpu.dp.cpp end====="
// RUN: FileCheck %S/bar/util.gpu --match-full-lines --input-file %T/bar/util.gpu.dp.cpp
// RUN: echo "=====util.gpuhead begin====="
// RUN: cat %T/bar/util.gpuhead
// RUN: echo "=====util.gpuhead end====="
// RUN: FileCheck %S/bar/util.gpuhead --match-full-lines --input-file %T/bar/util.gpuhead
// RUN: echo "=====util_bar.hh begin====="
// RUN: cat%T/bar/util_bar.hh
// RUN: echo "=====util_bar.hh end====="
// RUN: FileCheck %S/bar/util_bar.hh --match-full-lines --input-file %T/bar/util_bar.hh
// RUN: echo "=====macro_def.hh begin====="
// RUN: cat %T/bar/macro_def.hh
// RUN: echo "=====macro_def.hh end====="
// RUN: FileCheck %S/bar/macro_def.hh --match-full-lines --input-file %T/bar/macro_def.hh
// RUN: echo "=====util_bar.cc.dp.cpp begin====="
// RUN: cat %T/bar/util_bar.cc.dp.cpp
// RUN: echo "=====util_bar.cc.dp.cpp end====="
// RUN: FileCheck %S/bar/util_bar.cc --match-full-lines --input-file %T/bar/util_bar.cc.dp.cpp

// RUN: dpct --format-range=none --cuda-include-path="%cuda-path/include" -in-root=%S -out-root=%T -p=%T  %S/main.gpu   --sycl-named-lambda 
// RUN: echo "=====main.gpu.dp.cpp begin====="
// RUN: cat %T/main.gpu.dp.cpp
// RUN: echo "=====main.gpu.dp.cpp end====="
// RUN: FileCheck %S/main.gpu --match-full-lines --input-file %T/main.gpu.dp.cpp

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
