// RUN: :
#include "cuda_fp16.h"

// CHECK: sycl::half test(void) {
half test(void) {
  return half();
}
