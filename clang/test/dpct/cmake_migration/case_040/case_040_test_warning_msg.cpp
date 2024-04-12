// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.cmake ./input.cmake
// RUN: dpct -in-root ./ -out-root out  ./input.cmake --migrate-build-script-only > migration.log 2>&1

// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out/input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt

// CHECK: begin
// CHECK-NEXT: end

// grep "input.cmake:1:warning:DPCT3000:0: Migration of syntax \"cuda_select_nvcc_arch_flags\" is not supported. You may need to adjust the code." ./migration.log
