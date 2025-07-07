// UNSUPPORTED: system-windows
// RUN: echo "empty command"


#include <iostream>
#include <cuda_runtime.h>

// CHECK: void test_foo(){
__device__ void test_foo(void){
}
