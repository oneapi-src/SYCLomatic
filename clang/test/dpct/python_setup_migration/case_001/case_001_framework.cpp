// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.py ./input.py
// RUN: cp %S/input.cmake ./input.cmake
// RUN: cp %S/src/input.cu.txt ./input.cu
// RUN: cp %S/src/compile_commands.json ./compile_commands.json
// RUN: dpct -in-root ./ -out-root out --migrate-build-script-only

// RUN: echo "begin" > %T/diff_1.txt
// RUN: diff --strip-trailing-cr %S/expected.py %T/out/input.py >> %T/diff_1.txt
// RUN: echo "end" >> %T/diff_1.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: echo "begin" > %T/diff_2.txt
// RUN: diff --strip-trailing-cr %S/expected.cmake %T/out/input.cmake >> %T/diff_2.txt
// RUN: echo "end" >> %T/diff_2.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: dpct -in-root ./ -out-root out -p ./ -migrate-build-script=Python_Setup
// RUN: echo "begin" > %T/diff_3.txt
// RUN: diff --strip-trailing-cr %S/src/expected.cpp.txt %T/out/input.dp.cpp >> %T/diff_3.txt
// RUN: echo "end" >> %T/diff_3.txt
// CHECK: begin
// CHECK-NEXT: end
