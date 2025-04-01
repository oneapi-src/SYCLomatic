// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: mkdir -p out
// RUN: cp %S/input.cmake ./input.cmake
// RUN: cp %S/MainSourceFiles.yaml ./out/MainSourceFiles.yaml
// RUN: dpct -in-root ./ -out-root out  ./input.cmake --migrate-build-script-only
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out/input.cmake >> %T/diff.txt
