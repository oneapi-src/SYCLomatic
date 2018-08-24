// RUN: cu2sycl -out-root %T %s -passes "ErrorHandlingIfStmtRule" -- -w -x cuda --cuda-host-only 2>&1 | FileCheck --match-full-lines %s

int printf(const char *s, ...);
int fprintf(int, const char *s, ...);

// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1001: 'malloc(256)' couldn't be removed. See details in the resulting file comments
// CHECK:    malloc(0x100);
// CHECK:    ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten, see details in the resulting file comments
// CHECK:  if (err != cudaSuccess) {
// CHECK:  ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1001: 'x = printf("fmt string")' couldn't be removed. See details in the resulting file comments
// CHECK:    x = printf("fmt string");
// CHECK:    ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten, see details in the resulting file comments
// CHECK:  if (err)
// CHECK:  ^
void test_side_effects(cudaError_t err, int arg, int x, int y, int z) {
  if (err != cudaSuccess) {
    malloc(0x100);
    printf("error!\n");
    exit(1);
  }
  if (err)
    x = printf("fmt string");
}


// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1001: 'malloc(256)' couldn't be removed. See details in the resulting file comments
// CHECK:    malloc(0x100);
// CHECK:    ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten, see details in the resulting file comments
// CHECK:  if (err == cudaErrorAssert) {
// CHECK:  ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: note: SYCLCT1002: Special case error handling if-stmt was detected:'err == 255', you may need to rewrite this code
// CHECK:  if (err == 255) {
// CHECK:      ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1001: 'malloc(256)' couldn't be removed. See details in the resulting file comments
// CHECK:    malloc(0x100);
// CHECK:    ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten, see details in the resulting file comments
// CHECK:  if (err == 255) {
// CHECK:  ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: note: SYCLCT1002: Special case error handling if-stmt was detected:'err == 1', you may need to rewrite this code
// CHECK:  if (err == 1) {
// CHECK:      ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1001: 'malloc(256)' couldn't be removed. See details in the resulting file comments
// CHECK:    malloc(0x100);
// CHECK:    ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten, see details in the resulting file comments
// CHECK:  if (err == 1) {
// CHECK:  ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: note: SYCLCT1002: Special case error handling if-stmt was detected:'666 == err', you may need to rewrite this code
// CHECK:  if (666 == err) {
// CHECK:      ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1001: 'malloc(256)' couldn't be removed. See details in the resulting file comments
// CHECK:    malloc(0x100);
// CHECK:    ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten, see details in the resulting file comments
// CHECK:  if (666 == err) {
// CHECK:  ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: note: SYCLCT1002: Special case error handling if-stmt was detected:'cudaErrorAssert == err', you may need to rewrite this code
// CHECK:  if (cudaErrorAssert == err) {
// CHECK:      ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1001: 'malloc(256)' couldn't be removed. See details in the resulting file comments
// CHECK:    malloc(0x100);
// CHECK:    ^
// CHECK:{{[^:]+:[0-9]+:[0-9]+}}: warning: SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten, see details in the resulting file comments
// CHECK:  if (cudaErrorAssert == err) {
// CHECK:  ^
void specialize_ifs_negative() {
  cudaError_t err;
  if (err == cudaSuccess) {
    printf("efef");
  }
  if (err == cudaErrorAssert) {
    printf("efef");
    malloc(0x100);
  }
  if (err == 255) {
    malloc(0x100);
  }
  if (err == 1) {
    malloc(0x100);
  }
  if (666 == err) {
    malloc(0x100);
  }
  if (cudaErrorAssert == err) {
    malloc(0x100);
  }
}