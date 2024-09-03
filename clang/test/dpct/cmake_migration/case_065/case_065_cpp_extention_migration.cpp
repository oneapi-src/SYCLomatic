// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp -r %S/nvcv_types ./nvcv_types
// RUN: mkdir -p out
// RUN: cp %S/MainSourceFiles.yaml ./out
// RUN: dpct -in-root ./ -out-root out   --migrate-build-script-only

// RUN: echo "begin" > %T/diff_1.txt
// RUN: diff --strip-trailing-cr %S/CMakeLists_outer.ref %T/out/nvcv_types/CMakeLists.txt >> %T/diff_1.txt
// RUN: echo "end" >> %T/diff_1.txt
// CHECK: begin
// CHECK-NEXT: end

// RUN: echo "begin" > %T/diff_2.txt
// RUN: diff --strip-trailing-cr %S/CMakeLists_inner.ref %T/out/nvcv_types/priv/CMakeLists.txt >> %T/diff_2.txt
// RUN: echo "end" >> %T/diff_2.txt
// CHECK: begin
// CHECK-NEXT: end
