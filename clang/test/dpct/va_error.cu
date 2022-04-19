// RUN: dpct --format-range=none -in-root=%S -out-root %T/va_error %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/va_error/va_error.dp.cpp

#include "cuda_runtime.h"
#include <stdio.h>

// CHECK: void foo();
__global__ void foo();

static void test_va(const char *fmt,...)
{
    va_list ap;
    va_start(ap, fmt);
    va_end(ap);
}

