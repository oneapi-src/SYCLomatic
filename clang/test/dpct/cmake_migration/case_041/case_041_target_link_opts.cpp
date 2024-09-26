// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.cmake ./input.cmake
// RUN: dpct -in-root ./ -out-root out  ./input.cmake --migrate-build-script-only  > migration1.log 2>&1
// RUN: cat %S/ref.txt > ./check.txt
// RUN: cat migration1.log >> ./check.txt
// RUN: FileCheck --match-full-lines --input-file check.txt check.txt

// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out/input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt

// CHECK: begin
// CHECK-NEXT: end
