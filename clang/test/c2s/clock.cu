// RUN: c2s --format-range=none -out-root %T/clock %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
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
    // CHECK-NEXT: DPCT1008:{{[0-9]+}}: clock function is not defined in the DPC++. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
    // CHECK-NEXT: */
    *timer = clock();

    // CHECK: /*
    // CHECK-NEXT: DPCT1008:{{[0-9]+}}: clock function is not defined in the DPC++. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
    // CHECK-NEXT: */
    clock();

    // CHECK: /*
    // CHECK-NEXT: DPCT1008:{{[0-9]+}}: clock64 function is not defined in the DPC++. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
    // CHECK-NEXT: */
    *timer = clock64();

    // CHECK: /*
    // CHECK-NEXT: DPCT1008:{{[0-9]+}}: clock64 function is not defined in the DPC++. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
    // CHECK-NEXT: */
    clock64();
}

int main(int argc, char **argv)
{
    float *dinput = NULL;
    float *doutput = NULL;
    clock_t *dtimer = NULL;
    timedReduction<<<64, 256, sizeof(float) * 2 * 256>>>(dinput, doutput, dtimer);
    return 0;
}

