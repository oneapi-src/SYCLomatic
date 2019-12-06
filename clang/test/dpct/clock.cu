// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/clock.dp.cpp

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
}

int main(int argc, char **argv)
{
    float *dinput = NULL;
    float *doutput = NULL;
    clock_t *dtimer = NULL;

    timedReduction<<<64, 256, sizeof(float) * 2 * 256>>>(dinput, doutput, dtimer);

    return 0;
}
