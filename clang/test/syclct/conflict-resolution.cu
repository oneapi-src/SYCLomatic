// RUN: syclct -out-root %T %s -passes "ErrorHandlingIfStmtRule,ErrorConstantsRule" -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/conflict-resolution.sycl.cpp

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
