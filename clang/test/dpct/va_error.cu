// RUN: dpct -in-root=%S -out-root %T %s -- -std=c++14  -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/va_error.dp.cpp

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
