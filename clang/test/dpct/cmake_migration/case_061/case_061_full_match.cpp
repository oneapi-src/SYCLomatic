// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/input.cmake ./input.cmake
// RUN: cp %S/input.cmake ./input2.cmake
// RUN: dpct -in-root ./ -out-root out  ./input.cmake --migrate-build-script-only
// RUN: diff --strip-trailing-cr %S/expected.txt %T/out/input.cmake >> %T/diff.txt

// RUN: test  -f %T/out/input.cmake
// RUN: test  -f %T/out/dpct.cmake
// RUN: test ! -f %T/out/input2.cmake
