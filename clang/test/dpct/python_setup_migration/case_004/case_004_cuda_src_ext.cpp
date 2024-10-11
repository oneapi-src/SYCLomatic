// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.py ./input.py
// RUN: dpct -in-root ./ -out-root out ./input.py --migrate-build-script-only --rule-file=%T/../../../../../../../extensions/python_setup_rules/python_setup_script_migration_rule_ipex.yaml
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.py %T/out/input.py >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt

// CHECK: begin
// CHECK-NEXT: end
