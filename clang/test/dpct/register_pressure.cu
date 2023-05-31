// RUN: dpct --optimize-migration --format-range=none -out-root %T/register_pressure %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/register_pressure/register_pressure.dp.cpp
#include<cuda_runtime.h>

// CHECK: /*
// CHECK: DPCT1110:{{[0-9]+}}: The total declared local variable size in device function kernel exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
// CHECK: */
__global__ void kernel(){
  int a[100];
}

int main(void) {

    kernel<<<1, 1>>>();
    return 0;

}