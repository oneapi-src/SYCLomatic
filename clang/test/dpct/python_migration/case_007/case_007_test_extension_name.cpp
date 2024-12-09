// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: mkdir -p out
// RUN: cp %S/input.py ./input.py
// RUN: cp %S/MainSourceFiles.yaml ./out/MainSourceFiles.yaml
// RUN: dpct -in-root ./ -out-root out ./input.py --migrate-build-script-only
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.py %T/out/input.py >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt

// CHECK: begin
// CHECK-NEXT: end
