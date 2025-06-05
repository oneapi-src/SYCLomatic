// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.cmake ./input.cmake
// RUN: dpct -in-root ./ -out-root out  ./input.cmake --migrate-build-script-only --rule-file=%S/local_replace_implicit_migration_rule.yaml > migration.log 2>&1
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out/input.cmake >> %T/diff.txt

// RUN: cat %S/migration_ref.log > %T/check_migration_check.log
// RUN: cat %T/migration.log >>%T/check_migration_check.log
// RUN: FileCheck --match-full-lines --input-file %T/check_migration_check.log %T/check_migration_check.log
