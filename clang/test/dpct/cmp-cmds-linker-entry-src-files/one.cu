// UNSUPPORTED: system-windows
// intercept-build only supports Linux
//
// ------ prepare test directory
// RUN: cd %T
// RUN: cp %s one.cu
// RUN: cp %S/two.cu two.cu
// RUN: cp %S/three.c three.c
// RUN: cp %S/four.c four.c
// RUN: cp %S/Makefile Makefile
//
// ------ create reference compilation database
// RUN: echo "// CHECK: [" > compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 -o one.o -D__CUDACC__=1 one.cu\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"file\": \"%/T/one.cu\"" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    }," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 -o two.o -D__CUDACC__=1 two.cu\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"file\": \"%/T/two.cu\"" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    }," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 -D__CUDACC__=1 -o four.o four.c\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"file\": \"%/T/four.c\"" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    }," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 -D__CUDACC__=1 -o three.o three.c\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"file\": \"%/T/three.c\"" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    }," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc  -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 one.o two.o -o main -D__CUDACC__=1 four.o three.o\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T\"" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    }" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:]" >> compile_commands.json_ref
//
// ----- Test to use option 'intercept-build' and build command 'make'
// RUN: make clean
// RUN: not dpct intercept-build make
// RUN: FileCheck --match-full-lines --input-file %T/compile_commands.json %T/compile_commands.json_ref

#include <iostream>

int main() {

  return 0;
}
