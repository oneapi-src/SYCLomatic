// UNSUPPORTED: system-windows
// intercept-build only supports Linux
//
// ------ prepare test directory
// RUN: cd %T
// RUN: cp %s main.cpp
// RUN: cp -r %S/kernels kernels/
// RUN: cp %S/Makefile Makefile
//
// ------ create reference compilation database
// RUN: echo "// CHECK: [" > compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -o test.o -D__CUDACC__=1 kernels/test.cu\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"file\": \"%/T/kernels/test.cu\"" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    }," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  test.o -o all -D__CUDACC__=1 main.cpp\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T\"," >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:        \"file\": \"%/T/main.cpp\"" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:    }" >> compile_commands.json_ref
// RUN: echo "// CHECK-NEXT:]" >> compile_commands.json_ref
//
// ----- Test to use option 'intercept-build' and build command 'make'
// RUN: make clean
// RUN: not dpct intercept-build make
// RUN: FileCheck --match-full-lines --input-file %T/compile_commands.json %T/compile_commands.json_ref
//
// ----- Test to use option '-intercept-build' and build command 'make -B'
// RUN: make clean
//        The 'not' prefix is used to invert the dpct return code, and thus prevent a LIT failure.
//        dpct returns non-zero because input has no CUDA code, but the return-code can be ignored
//        because the code is not relevant to this testcase.
// RUN: not dpct -intercept-build make -B
// RUN: FileCheck --match-full-lines --input-file %T/compile_commands.json %T/compile_commands.json_ref
//
// ----- Test to use option '--intercept-build -cdb' and build command 'make -B'
// RUN: make clean
// RUN: not dpct --intercept-build -vvv --cdb compile_commands2.json make -B > intc_build_output.txt
// RUN: grep "verbose=3" intc_build_output.txt > check_verbose_flag.txt
// RUN: echo "// CHECK: verbose=3" > check_verbose_flag_ref.txt
// RUN: FileCheck --input-file %T/check_verbose_flag.txt %T/check_verbose_flag_ref.txt
// RUN: FileCheck --match-full-lines --input-file %T/compile_commands2.json %T/compile_commands.json_ref

// ------ cleanup test directory
// RUN: cd ..
// RUN: rm -rf ./dpct-intercept-build

#include <stdio.h>
#include "kernels/test.cuh"

int main() {
	  wrap_test_print();
	    return 0;
}
