// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.py ./input.py

// RUN: dpct -in-root ./ -out-root out_pytorch ./input.py --migrate-build-script-only --migrate-build-script=Python --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_pytorch.yaml
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.py %T/out_pytorch/input.py >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: dpct -in-root ./ -out-root out_ipex ./input.py --migrate-build-script-only --migrate-build-script=Python --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_ipex.yaml
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.py %T/out_ipex/input.py >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// CHECK: begin
// CHECK-NEXT: end
