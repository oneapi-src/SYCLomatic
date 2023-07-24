// UNSUPPORTED: system-windows
// intercept-build not found on windows
// ------ prepare test directory
// RUN: cd %T
// RUN: rm -rf compile_commands_test
// RUN: mkdir  compile_commands_test
// RUN: cd     compile_commands_test
// RUN: cp %s %T/compile_commands_test/one.cu
// RUN: cp %S/two.cu %T/compile_commands_test/two.cu
// RUN: cp %S/three.c %T/compile_commands_test/three.c
// RUN: cp %S/four.c %T/compile_commands_test/four.c
// RUN: cp %S/Makefile %T/compile_commands_test/Makefile


// RUN: echo "// CHECK: [" > compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 -o one.o -D__CUDACC__=1 one.cu\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T/compile_commands_test\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:       \"file\": \"%/T/compile_commands_test/one.cu\"" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:   }," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 -o two.o -D__CUDACC__=1 two.cu\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T/compile_commands_test\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:       \"file\": \"%/T/compile_commands_test/two.cu\"" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:   }," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 -D__CUDACC__=1 -o four.o four.c\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T/compile_commands_test\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:       \"file\": \"%/T/compile_commands_test/four.c\"" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:   }," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 -D__CUDACC__=1 -o three.o three.c\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T/compile_commands_test\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:       \"file\": \"%/T/compile_commands_test/three.c\"" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:   }," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -std=c++14 -Wall --cuda-gpu-arch=sm_60 -O3 one.o two.o -o main -D__CUDACC__=1 four.o three.o\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T/compile_commands_test\"" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:   }" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:]" >> compile_commands.json.ref

// RUN: intercept-build make -f Makefile
// RUN: FileCheck --match-full-lines --input-file %T/compile_commands_test/compile_commands.json %T/compile_commands_test/compile_commands.json.ref 

#include <iostream>

int main() {

  return 0;
}
