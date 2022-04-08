// RUN: c2s --format-range=none -out-root %T/conflict-resolution %s -passes "ErrorHandlingIfStmtRule,ErrorConstantsRule" --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/conflict-resolution/conflict-resolution.dp.cpp

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

