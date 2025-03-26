// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp -r %S/nvcv_types ./nvcv_types
// RUN: mkdir -p out
// RUN: cp %S/MainSourceFiles.yaml ./out
// RUN: dpct -in-root ./ -out-root out   --migrate-build-script-only

// RUN: diff --strip-trailing-cr %S/CMakeLists_outer.ref %T/out/nvcv_types/CMakeLists.txt >> %T/diff.txt

// RUN: diff --strip-trailing-cr %S/CMakeLists_inner.ref %T/out/nvcv_types/priv/CMakeLists.txt >> %T/diff.txt
