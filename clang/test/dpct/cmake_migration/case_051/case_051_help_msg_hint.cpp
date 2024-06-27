// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: dpct -in-root ./ -out-root out  --migrate-build-script-only > migration1.log 2>&1
// RUN: cat %S/ref.txt > ./check.txt
// RUN: cat migration1.log >> ./check.txt
// RUN: FileCheck --match-full-lines --input-file check.txt check.txt

void foo() {}
