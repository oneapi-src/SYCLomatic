// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.cmake ./input.cmake
// RUN: dpct -in-root ./ -out-root out  ./input.cmake --migrate-build-script-only --cuda-include-path="%cuda-path/include" --rule-file=%T/../../../../../../../extensions/cmake_rules/cmake_script_migration_rule_optional.yaml
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out/input.cmake >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt

// CHECK: begin
// CHECK-NEXT: end
