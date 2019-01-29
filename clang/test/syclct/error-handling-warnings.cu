// RUN: syclct -out-root %T %s  -- -w -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/error-handling-warnings.sycl.cpp

int printf(const char *s, ...);
int fprintf(int, const char *s, ...);

// CHECK:void test_side_effects(int err, int arg, int x, int y, int z) try {
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1000:{{[0-9]+}}: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err != 0) {
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001:{{[0-9]+}}: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:    exit(1);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1000:{{[0-9]+}}: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err)
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001:{{[0-9]+}}: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
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
// CHECK:void specialize_ifs_negative() try {
// CHECK-NEXT:  int err;
// CHECK-NEXT:  if (err == 0) {
// CHECK-NEXT:    printf("efef");
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. It couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder> 
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err == 59) {
// CHECK-NEXT:    printf("efef");
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001:{{[0-9]+}}: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. It couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err == 255) {
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001:{{[0-9]+}}: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. It couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err == 1) {
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001:{{[0-9]+}}: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. It couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (666 == err) {
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001:{{[0-9]+}}: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1002:{{[0-9]+}}: Special case error handling if-stmt was detected. It couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (59 == err) {
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001:{{[0-9]+}}: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
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
