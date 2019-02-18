// RUN: syclct -out-root %T %s -- -std=c++11  -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/clock.sycl.cpp

#include <stdio.h>
#include <stdint.h>

__global__ static void timedReduction(const float *input, float *output, clock_t *timer)
{
    // CHECK: /*
    // CHECK-NEXT: SYCLCT1008:{{[0-9]+}}: Function clock is not defined in the SYCL specification. This is a hardware-specific feature. Consider consulting with hardware vendor to find a replacement.
    // CHECK-NEXT: */
    *timer = clock();

    // CHECK: /*
    // CHECK-NEXT: SYCLCT1008:{{[0-9]+}}: Function clock is not defined in the SYCL specification. This is a hardware-specific feature. Consider consulting with hardware vendor to find a replacement.
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
