// RUN: syclct -out-root %T %s -passes "ErrorHandlingIfStmtRule" -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: sed -e 's,//.*$,,' %T/error-handling.sycl.cpp | FileCheck --match-full-lines %s

int printf(const char *s, ...);
int fprintf(int, const char *s, ...);

// CHECK:void test_simple_ifs() {
// CHECK-NEXT:  cudaError_t err;
// checking for empty lines (with one or more spaces)
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:}
void test_simple_ifs() {
  cudaError_t err;
  if (err != cudaSuccess) {
  }
  if (err) {
  }
  if (err != 0) {
  }
  if (0 != err) {
  }
  if (cudaSuccess != err) {
  }
  if (err != cudaSuccess) {
  }
}

// CHECK:void test_simple_ifs_const() {
// CHECK-NEXT:  const cudaError_t err = cudaSuccess;
// Checking for empty lines (with one or more spaces).
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:}
void test_simple_ifs_const() {
  const cudaError_t err = cudaSuccess;
  if (err != cudaSuccess) {
  }
  if (err) {
  }
  if (err != 0) {
  }
  if (0 != err) {
  }
  if (cudaSuccess != err) {
  }
  if (err != cudaSuccess) {
  }
}

// CHECK:void test_typedef() {
// CHECK-NEXT:  typedef cudaError_t someError_t;
// CHECK-NEXT:  someError_t err;
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:}
void test_typedef() {
  typedef cudaError_t someError_t;
  someError_t err;
  if (err != cudaSuccess) {
  }
  if (0 != err) {
  }
}

// CHECK:void test_no_braces() {
// CHECK-NEXT:  cudaError_t err;
// CHECK-NEXT:  {{ +}};
// CHECK-NEXT:}
void test_no_braces() {
  cudaError_t err;
  if (err != cudaSuccess)
    printf("error!\n");
}

// CHECK:void test_unrelated_then() {
// CHECK-NEXT:  cudaError_t err;
// CHECK-NEXT:  int i = 0;
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err != cudaSuccess) {
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:    ++i;
// CHECK-NEXT:  }
// CHECK-NEXT:}

void test_unrelated_then() {
  cudaError_t err;
  int i = 0;
  if (err != cudaSuccess) {
    ++i;
  }
}

// CHECK:void test_CUDA_SUCCESS() {
// CHECK-NEXT:  cudaError_t err;
// CHECK-NEXT:  {{ +}}
// CHECK-NEXT:}
void test_CUDA_SUCCESS() {
  cudaError_t err;
  if (err != CUDA_SUCCESS) {
    printf("error!\n");
  }
}

// CHECK:void test_CUDA_SUCCESS_empty() {
// CHECK-NEXT:  cudaError_t err;
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:}
void test_CUDA_SUCCESS_empty() {
  cudaError_t err;
  if (err != CUDA_SUCCESS) {
  }
}

// CHECK:void test_other_enum() {
// CHECK-NEXT:  cudaError_t err;
// CHECK-NEXT:  if (err != cudaErrorLaunchFailure) {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_other_enum() {
  cudaError_t err;
  if (err != cudaErrorLaunchFailure) {
    printf("error!\n");
  }
}

// CHECK:void test_assignment() {
// CHECK-NEXT:  cudaError_t err;
// CHECK-NEXT:  if (err = cudaMalloc(0, 0)) {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_assignment() {
  cudaError_t err;
  if (err = cudaMalloc(0, 0)) {
    printf("error!\n");
  }
}

// CHECK:void test_1(cudaError_t err, int arg) {
// CHECK-NEXT:  if (err == cudaSuccess && arg) {
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_1(cudaError_t err, int arg) {
  if (err == cudaSuccess && arg) {
  }
}

// CHECK:void test_12(cudaError_t err, int arg) {
// CHECK-NEXT:  if (err) {
// CHECK-NEXT:  } else {
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_12(cudaError_t err, int arg) {
  if (err) {
  } else {
    
  }
}

// CHECK:void test_13(cudaError_t err, int arg) {
// CHECK-NEXT:  {{ +}}
// CHECK-NEXT:}
void test_13(cudaError_t err, int arg) {
  if (err) {
    printf("error!\n");
  }
}

// CHECK:void test_14(cudaError_t err, int arg) {
// CHECK-NEXT:  if (arg == 1) {
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
// CHECK-NEXT  if (arg != 0) {
// CHECK-NEXT    return;
// CHECK-NEXT  }
// CHECK-NEXT  if (arg) {
// CHECK-NEXT    return;
// CHECK-NEXT  }
// CHECK-NEXT}
void test_14(cudaError_t err, int arg) {
  if (arg == 1) {
    return;
  }
  if (arg != 0) {
    return;
  }
  if (arg) {
    return;
  }
}

// CHECK:void test_15(cudaError_t err, int arg) {
// CHECK-NEXT:  if (cudaMalloc(0, 0)) {
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_15(cudaError_t err, int arg) {
  if (cudaMalloc(0, 0)) {
  }
}

// CHECK:void test_16(cudaError_t err, int arg) {
// CHECK-NEXT:  if (err) {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:    exit(1);
// CHECK-NEXT:  } else {
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_16(cudaError_t err, int arg) {
  if (err) {
    printf("error!\n");
    exit(1);
  } else {
    
  }
}

// CHECK:void test_17(cudaError_t err, int arg) {
// CHECK-NEXT:  if (!cudaMalloc(0, 0)) {
// CHECK-NEXT:  } else {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:    exit(1);
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_17(cudaError_t err, int arg) {
  if (!cudaMalloc(0, 0)) {
  } else {
    printf("error!\n");
    exit(1);
  }
}

// CHECK:void test_18(cudaError_t err, int arg) {
// CHECK-NEXT:  if (err)
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:  else
// CHECK-NEXT:    printf("success!\n");
// CHECK-NEXT:}
void test_18(cudaError_t err, int arg) {
  if (err)
    printf("error!\n");
  else
    printf("success!\n");
}

// CHECK:void test_19(cudaError_t err, int arg) {
// CHECK-NEXT:  if (err && arg) {
// CHECK-NEXT:  } else {
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_19(cudaError_t err, int arg) {
  if (err && arg) {
  } else {
  }
}

// CHECK:void test_compare_to_3(cudaError_t err, int arg) {
// CHECK-NEXT:  if (err != 3) {
// CHECK-NEXT:  }
// CHECK-NEXT:}
void test_compare_to_3(cudaError_t err, int arg) {
  if (err != 3) {
  }
}

// CHECK:void test_21(const cudaError_t& err, int arg) {
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:}
void test_21(const cudaError_t& err, int arg) {
  if (err != 0) {
  }
}

// CHECK:void test_no_side_effects(cudaError_t err, int arg) {
// CHECK-NEXT: ;
// CHECK-NEXT: ;
// CHECK-NEXT: ;
// CHECK-NEXT:  {{ +}}
// CHECK-NEXT:}
void test_no_side_effects(cudaError_t err, int arg) {
  if (err)
    printf("efef");
  if (err)
    fprintf(0, "efef");
  if (err)
    exit(1);
  if (err != cudaSuccess) {
    printf("error!\n");
    exit(1);
  }
}

// CHECK:void test_side_effects(cudaError_t err, int arg, int x, int y, int z) {
// CHECK-NEXT:  ;
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err)
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err != cudaSuccess) {
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:    malloc(0x100);
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:    exit(1);
// CHECK-NEXT:  }
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err)
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1001: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK-NEXT:*/
// CHECK-NEXT:    x = printf("fmt string");
// CHECK-NEXT:  ;
// CHECK-NEXT:}

void test_side_effects(cudaError_t err, int arg, int x, int y, int z) {
  if (err)
    printf("efef %i", malloc(0x100));
  if (err)
    malloc(0x100);
  if (err != cudaSuccess) {
    malloc(0x100);
    printf("error!\n");
    exit(1);
  }
  if (err)
    x = printf("fmt string");
  if (err)
    printf("fmt string %d", y + z);
}

// CHECK:void specialize_ifs() {
// CHECK-NEXT:  cudaError_t err;
// checking for empty lines (with one or more spaces)
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:{{ +}}
// CHECK-NEXT:}
void specialize_ifs() {
  cudaError_t err;
  if (err == cudaErrorAssert) {
    printf("efef");
  }
  if (err == 255) {
  }
  if (err == 1) {
  }
  if (666 == err) {
  }
  if (cudaErrorAssert == err) {
  }
}

// CHECK:void specialize_ifs_negative() {
// CHECK:  cudaError_t err;
// CHECK:  if (err == cudaSuccess) {
// CHECK:    printf("efef");
// CHECK:  }
// CHECK:/*
// CHECK:SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:  if (err == cudaErrorAssert) {
// CHECK:    printf("efef");
// CHECK:/*
// CHECK:SYCLCT1001: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:    malloc(0x100);
// CHECK:  }
// CHECK:/*
// CHECK:SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:  if (err == 255) {
// CHECK:/*
// CHECK:SYCLCT1001: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:    malloc(0x100);
// CHECK:  }
// CHECK:/*
// CHECK:SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:  if (err == 1) {
// CHECK:/*
// CHECK:SYCLCT1001: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:    malloc(0x100);
// CHECK:  }
// CHECK:/*
// CHECK:SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:  if (666 == err) {
// CHECK:/*
// CHECK:SYCLCT1001: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:    malloc(0x100);
// CHECK:  }
// CHECK:/*
// CHECK:SYCLCT1000: Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:  if (cudaErrorAssert == err) {
// CHECK:/*
// CHECK:SYCLCT1001: Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>
// CHECK:*/
// CHECK:    malloc(0x100);
// CHECK:  }
// CHECK:}
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
