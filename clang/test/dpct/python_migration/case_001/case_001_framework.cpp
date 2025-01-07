// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.py ./input.py
// RUN: cp %S/input.cmake ./input.cmake
// RUN: cp %S/src/input.cu.txt ./input.cu
// RUN: cp %S/src/compile_commands.json ./compile_commands.json

// RUN: dpct -in-root ./ -out-root out_pytorch_1 --migrate-build-script-only --migrate-build-script=Python --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_pytorch.yaml
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected_pytorch.py %T/out_pytorch_1/input.py >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: dpct -in-root ./ -out-root out_pytorch_2 --migrate-build-script-only --migrate-build-script=CMake,Python --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_pytorch.yaml
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected_pytorch.py %T/out_pytorch_2/input.py >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.cmake %T/out_pytorch_2/input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: dpct -in-root ./ -out-root out_ipex_1 --cuda-include-path="%cuda-path/include" --migrate-build-script=Python --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_ipex.yaml -p ./
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected_ipex.py %T/out_ipex_1/input.py >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/src/expected.cpp.txt %T/out_ipex_1/input.dp.cpp >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: dpct -in-root ./ -out-root out_ipex_2 --cuda-include-path="%cuda-path/include" --migrate-build-script=CMake,Python --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_ipex.yaml -p ./
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected_ipex.py %T/out_ipex_2/input.py >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.cmake %T/out_ipex_2/input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/src/expected.cpp.txt %T/out_ipex_2/input.dp.cpp >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end
