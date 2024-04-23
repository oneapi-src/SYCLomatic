// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.cmake ./input.cmake
// RUN: mkdir -p subdir
// RUN: cp %S/sub_input.cmake ./subdir/sub_input.cmake
// RUN: dpct -in-root ./ -out-root out  ./input.cmake ./subdir/sub_input.cmake --migrate-build-script-only
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out/input.cmake >> %T/diff.txt
// RUN: diff --strip-trailing-cr %S/sub_expected.txt %T/out/subdir/sub_input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt

// CHECK: begin
// CHECK-NEXT: end

// The command below is used to test if dpct.cmake has been write to the output directory
// RUN: diff --strip-trailing-cr %T/out/dpct.cmake %T/out/dpct.cmake
