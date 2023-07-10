// RUN: dpct --format-range=none -out-root %T/clock %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/clock/clock.dp.cpp

// CHECK: #include <stdio.h>
// CHECK-NEXT: #include <stdint.h>
// CHECK-NEXT: #include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

__global__ static void timedReduction(const float *input, float *output, clock_t *timer)
{
    // CHECK: /*
    // CHECK-NEXT: DPCT1008:{{[0-9]+}}: clock function is not defined in SYCL. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
    // CHECK-NEXT: */
    *timer = clock();

    // CHECK: /*
    // CHECK-NEXT: DPCT1008:{{[0-9]+}}: clock function is not defined in SYCL. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
    // CHECK-NEXT: */
    clock();

    // CHECK: /*
    // CHECK-NEXT: DPCT1008:{{[0-9]+}}: clock64 function is not defined in SYCL. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
    // CHECK-NEXT: */
    *timer = clock64();

    // CHECK: /*
    // CHECK-NEXT: DPCT1008:{{[0-9]+}}: clock64 function is not defined in SYCL. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
    // CHECK-NEXT: */
    clock64();
}

// CHECK: void timedAddition(const float *input, float *output, clock_t *timer){
// CHECK-NEXT:    *timer = clock();
// CHECK-NEXT:    clock();
// CHECK-NEXT:}
void timedAddition(const float *input, float *output, clock_t *timer){
    *timer = clock();
    clock();
}

int main(int argc, char **argv)
{
    float *dinput = NULL;
    float *hinput = NULL;
    float *doutput = NULL;
    float *houtput = NULL;
    clock_t *dtimer = NULL;
    clock_t *htimer = NULL;
    timedReduction<<<64, 256, sizeof(float) * 2 * 256>>>(dinput, doutput, dtimer);
    timedAddition(hinput, houtput, htimer);
    return 0;
}
