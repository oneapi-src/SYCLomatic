// UNSUPPORTED: system-windows
// intercept-build not found on windows
// ------ prepare test directory
// RUN: cd %T
// RUN: rm -rf compile_commands_c_opt_test
// RUN: mkdir  compile_commands_c_opt_test
// RUN: cd     compile_commands_c_opt_test

// RUN: cp %s %T/compile_commands_c_opt_test/one.cu
// RUN: cp %S/two.cu %T/compile_commands_c_opt_test/two.cu
// RUN: cp %S/three.cu %T/compile_commands_c_opt_test/three.cu
// RUN: cp %S/four.cu %T/compile_commands_c_opt_test/four.cu
// RUN: cp %S/Makefile %T/compile_commands_c_opt_test/Makefile

// RUN: echo "// CHECK: [" > compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc  -O2 -DKERNEL_USE_PROFILE=1 -DRUN_ON_GPU=1 -o one -allow-unsupported-compiler -D__CUDACC__=1 one.cu\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T/compile_commands_c_opt_test\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:       \"file\": \"%/T/compile_commands_c_opt_test/one.cu\"" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:   }," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -O2 -DKERNEL_USE_PROFILE=1 -DRUN_ON_GPU=1 -D__CUDACC__=1 two.cu\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T/compile_commands_c_opt_test\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:       \"file\": \"%/T/compile_commands_c_opt_test/two.cu\"" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:   }," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -O2 -DKERNEL_USE_PROFILE=1 -DRUN_ON_GPU=1 -D__CUDACC__=1 three.cu\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T/compile_commands_c_opt_test\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:       \"file\": \"%/T/compile_commands_c_opt_test/three.cu\"" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:   }," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:    {" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"command\": \"nvcc -c  -O2 -DKERNEL_USE_PROFILE=1 -DRUN_ON_GPU=1 -D__CUDACC__=1 four.cu\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:        \"directory\": \"%/T/compile_commands_c_opt_test\"," >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:       \"file\": \"%/T/compile_commands_c_opt_test/four.cu\"" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:   }" >> compile_commands.json.ref
// RUN: echo "// CHECK-NEXT:]" >> compile_commands.json.ref

// RUN: intercept-build make -f Makefile CC=nvcc USE_SM=70 -B
// RUN: FileCheck --match-full-lines --input-file %T/compile_commands_c_opt_test/compile_commands.json %T/compile_commands_c_opt_test/compile_commands.json.ref 

int main(){
    return 0;
}
