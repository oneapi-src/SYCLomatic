// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.cmake ./input.cmake
// RUN: dpct -in-root ./ -out-root out  ./input.cmake --migrate-build-script-only --rule-file=%S/duplicate_rule.yaml > migration.log 2>&1
// RUN: cat %S/ref.txt > ./check.txt
// RUN: cat migration.log >> ./check.txt
// RUN: FileCheck --match-full-lines --input-file check.txt check.txt