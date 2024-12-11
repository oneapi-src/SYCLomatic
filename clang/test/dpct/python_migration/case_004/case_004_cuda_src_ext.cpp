// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.py ./input.py

// RUN: dpct -in-root ./ -out-root out_pytorch ./input.py --migrate-build-script-only
// RUN: echo "begin" > %T/diff_pytorch.txt
// RUN: diff --strip-trailing-cr %S/expected.py %T/out_pytorch/input.py >> %T/diff_pytorch.txt
// RUN: echo "end" >> %T/diff_pytorch.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: dpct -in-root ./ -out-root out_ipex ./input.py --migrate-build-script-only --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_ipex.yaml
// RUN: echo "begin" > %T/diff_ipex.txt
// RUN: diff --strip-trailing-cr %S/expected.py %T/out_ipex/input.py >> %T/diff_ipex.txt
// RUN: echo "end" >> %T/diff_ipex.txt
// CHECK: begin
// CHECK-NEXT: end
