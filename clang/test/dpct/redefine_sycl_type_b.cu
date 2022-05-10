// RUN: echo pass
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "redefine_sycl_type.h"

void foo2() {
  float2 f2;
}
