// RUN: cu2sycl -out-root %T %s -passes "ErrorHandlingIfStmtRule,ErrorConstantsRule" -- -x cuda --cuda-host-only
// RUN: sed -e 's,//.*$,,' %T/conflict-resolution.sycl.cpp | FileCheck --match-full-lines %s

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
