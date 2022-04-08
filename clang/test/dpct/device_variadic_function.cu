// RUN: c2s --format-range=none -out-root %T/device_variadic_function %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device_variadic_function/device_variadic_function.dp.cpp
#include <cuda_runtime.h>
#include <cstdarg>
#include <cstdio>

// CHECK: /*
// CHECK-NEXT: DPCT1080:{{[0-9]+}}: Variadic functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: int variadic_prototype(int pa...){
__device__ int variadic_prototype(int pa...){

va_list args;
va_start(args, pa);
int b = va_arg(args, int);
va_end(args);
return b;
};

__global__ void func(int *p){
int a = *p;
int c = variadic_prototype(a,2,3);
printf("%d\n", c);
}

int main() {
int *a;
cudaMalloc(&a, 100);
func<<<1,1>>>(a);
cudaDeviceSynchronize();
return 0;

}
