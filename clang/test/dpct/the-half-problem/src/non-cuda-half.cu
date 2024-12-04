// RUN: :
#include "simple_half.h"

//CHECK: half test(float val) {
//CHECK-NEXT:   half tmp;
half test(float val) {
  half tmp;

  tmp.setBits(val);
  return tmp.bits();
}

//CHECK: void dummy() {}
__global__ void dummy() {}
