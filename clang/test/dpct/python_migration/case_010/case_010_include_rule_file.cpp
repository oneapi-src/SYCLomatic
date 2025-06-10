// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.py ./
// RUN: cp %S/expected.py ./
// RUN: cp -r %S/rules ./
// RUN: echo "!include %T/rules/b/b5.yaml" >> %T/rules/python_build_script_migration_rule_inc.yaml

// RUN: dpct -in-root ./ --migrate-build-script-only --migrate-build-script=Python --rule-file=%T/rules/python_build_script_migration_rule_inc.yaml input.py
// RUN: diff --strip-trailing-cr %S/expected.py %T/dpct_output/input.py
