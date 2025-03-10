// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.cmake ./input.cmake
// RUN: cp %S/input.cmake ./input2.cmake
// RUN: dpct -in-root ./ -out-root out  ./input.cmake --migrate-build-script-only
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out/input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt

// CHECK: begin
// CHECK-NEXT: end

// RUN: test  -f %T/out/input.cmake
// RUN: test  -f %T/out/dpct.cmake
// RUN: test ! -f %T/out/input2.cmake
