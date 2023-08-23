// RUN: echo "0"
#include "test_device.h"
#include "stdio.h"
__device__ void test_device() {
    printf("0000");
}
