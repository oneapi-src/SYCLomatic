// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.cmake ./input.cmake
// RUN: mkdir -p subdir
// RUN: cp %S/sub_input.cmake ./subdir/sub_input.cmake
// RUN: cp %S/src/input.cu.txt ./input.cu
// RUN: cp %S/src/compile_commands.json ./compile_commands.json

// RUN: dpct -in-root ./ -out-root out_1 ./input.cmake ./subdir/sub_input.cmake --migrate-build-script-only
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out_1/input.cmake >> %T/diff.txt
// RUN: diff --strip-trailing-cr %S/sub_expected.txt %T/out_1/subdir/sub_input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: dpct -in-root ./ -out-root out_2 ./input.cmake ./subdir/sub_input.cmake --migrate-build-script-only --migrate-build-script=CMake
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out_2/input.cmake >> %T/diff.txt
// RUN: diff --strip-trailing-cr %S/sub_expected.txt %T/out_2/subdir/sub_input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: dpct -in-root ./ -out-root out_3 --cuda-include-path="%cuda-path/include" --migrate-build-script=CMake -p ./
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out_3/input.cmake >> %T/diff.txt
// RUN: diff --strip-trailing-cr %S/sub_expected.txt %T/out_3/subdir/sub_input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/src/expected.cpp.txt %T/out_3/input.dp.cpp >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end

// The command below is used to test if dpct.cmake has been write to the output directory
// RUN: diff --strip-trailing-cr %T/out_1/dpct.cmake %T/out_1/dpct.cmake
// RUN: diff --strip-trailing-cr %T/out_2/dpct.cmake %T/out_2/dpct.cmake
// RUN: diff --strip-trailing-cr %T/out_3/dpct.cmake %T/out_3/dpct.cmake
