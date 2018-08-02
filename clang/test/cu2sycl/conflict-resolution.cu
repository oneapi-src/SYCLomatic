// RUN: cp %s %t
// RUN: cu2sycl %t -passes "ErrorHandlingIfStmtRule,ErrorConstantsRule" -- -x cuda --cuda-host-only
// RUN: sed -e 's,//.*$,,' %t | FileCheck --match-full-lines %s

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
