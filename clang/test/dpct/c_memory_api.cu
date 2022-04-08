// RUN: c2s --format-range=none -out-root %T/c_memory_api %s --cuda-include-path="%cuda-path/include" --extra-arg="-xc"
// RUN: FileCheck %s --match-full-lines --input-file %T/c_memory_api/c_memory_api.dp.cpp

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct new_s {
  float *fp;
  int *inp;
} new_t;

int main() {
  //CHECK:sycl::float2 *f2;
  //CHECK-NEXT:sycl::double3 *d3;
  //CHECK-NEXT:sycl::int4 *i4;
  //CHECK-NEXT:new_t *new_o;
  //CHECK-NEXT:new_o = (new_t *)calloc(1, sizeof(new_t));
  //CHECK-NEXT:f2 = (sycl::float2 *)realloc(new_o, sizeof(new_t));
  //CHECK-NEXT:d3 = (sycl::double3 *)malloc(sizeof(new_t));
  //CHECK-NEXT:i4 = (sycl::int4 *)calloc(1, sizeof(new_t));
  float2 *f2;
  double3 *d3;
  int4 *i4;
  new_t *new_o;
  new_o = calloc(1, sizeof(new_t));
  f2 = realloc(new_o, sizeof(new_t));
  d3 = malloc(sizeof(new_t));
  i4 = calloc(1, sizeof(new_t));
  return 0;
}

