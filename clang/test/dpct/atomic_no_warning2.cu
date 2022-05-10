// RUN: cd .
#include <cuda_runtime.h>
#include "atomic_no_warning.cuh"

__device__ void foo(){
  int a, b;
}