// RUN: dpct --format-range=none -out-root %T/conflict-resolution %s -passes "ErrorHandlingIfStmtRule,ErrorConstantsRule" --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/conflict-resolution/conflict-resolution.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/conflict-resolution/conflict-resolution.dp.cpp -o %T/conflict-resolution/conflict-resolution.dp.o %}

#ifndef NO_BUILD_TEST
int printf(const char *format, ...);

// CHECK: void test_00(cudaError_t err) {
// CHECK-NEXT:   {{ +}}
// CHECK-NEXT: }
void test_00(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("Some error happenned\n");
    exit(1);
  }
}
#endif
