// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.py ./input.py

// RUN: dpct -in-root ./ -out-root out_pytorch ./input.py --migrate-build-script-only --migrate-build-script=Python --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_pytorch.yaml
// RUN: diff --strip-trailing-cr %S/expected_pytorch.py %T/out_pytorch/input.py >> %T/diff.txt

// RUN: dpct -in-root ./ -out-root out_ipex ./input.py --migrate-build-script-only --migrate-build-script=Python --rule-file=%T/../../../../../../../extensions/python_rules/python_build_script_migration_rule_ipex.yaml
// RUN: diff --strip-trailing-cr %S/expected_ipex.py %T/out_ipex/input.py >> %T/diff.txt
