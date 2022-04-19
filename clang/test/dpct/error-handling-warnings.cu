// RUN: dpct --format-range=none -out-root %T/error-handling-warnings %s --cuda-include-path="%cuda-path/include" -- -w -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/error-handling-warnings/error-handling-warnings.dp.cpp

int printf(const char *s, ...);
int fprintf(int, const char *s, ...);

// CHECK:void test_side_effects(int err, int arg, int x, int y, int z) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1000:{{[0-9]+}}: Error handling if-stmt was detected but could not be rewritten.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err != 0) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:    exit(1);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1000:{{[0-9]+}}: Error handling if-stmt was detected but could not be rewritten.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err)
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    x = printf("fmt string");
// CHECK-NEXT:}

void test_side_effects(cudaError_t err, int arg, int x, int y, int z) {
  if (err != cudaSuccess) {
    malloc(0x100);
    printf("error!\n");
    exit(1);
  }
  if (err)
    x = printf("fmt string");
}
// CHECK:void specialize_ifs_negative() {
// CHECK-NEXT:  int err;
// CHECK-NEXT:  if (err == 0) {
// CHECK-NEXT:    printf("efef");
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to rewrite this code. 
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err == {{[0-9]+}}) {
// CHECK-NEXT:    printf("efef");
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err == 255) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err == 1) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (666 == err) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if ({{[0-9]+}} == err) {
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1001:{{[0-9]+}}: The statement could not be removed.
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:}
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

