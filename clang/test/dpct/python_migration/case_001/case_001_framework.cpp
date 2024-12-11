// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.py ./input.py
// RUN: cp %S/input.cmake ./input.cmake
// RUN: cp %S/src/input.cu.txt ./input.cu
// RUN: cp %S/src/compile_commands.json ./compile_commands.json

// RUN: dpct -in-root ./ -out-root out --migrate-build-script-only
// RUN: echo "begin" > %T/diff_pytorch.txt
// RUN: diff --strip-trailing-cr %S/expected_pytorch.py %T/out/input.py >> %T/diff_pytorch.txt
// RUN: echo "end" >> %T/diff_pytorch.txt
// CHECK: begin
// CHECK-NEXT: end
// RUN: echo "begin" > %T/diff_cmake_pytorch.txt
// RUN: diff --strip-trailing-cr %S/expected.cmake %T/out/input.cmake >> %T/diff_cmake_pytorch.txt
// RUN: echo "end" >> %T/diff_cmake_pytorch.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: dpct -in-root ./ -out-root out_ipex --cuda-include-path="%cuda-path/include" --migrate-build-script=Python --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_ipex.yaml -p ./
// RUN: echo "begin" > %T/diff_ipex.txt
// RUN: diff --strip-trailing-cr %S/expected_ipex.py %T/out_ipex/input.py >> %T/diff_ipex.txt
// RUN: echo "end" >> %T/diff_ipex.txt
// CHECK: begin
// CHECK-NEXT: end
// RUN: echo "begin" > %T/diff_cmake_ipex.txt
// RUN: diff --strip-trailing-cr %S/src/expected.cpp.txt %T/out_ipex/input.dp.cpp >> %T/diff_cmake_ipex.txt
// RUN: echo "end" >> %T/diff_cmake_ipex.txt
// CHECK: begin
// CHECK-NEXT: end
