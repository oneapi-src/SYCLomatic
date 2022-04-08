// RUN: echo
#include "common.h"

__global__ void foo_a(){
    foo<5>();
}