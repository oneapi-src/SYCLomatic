// RUN: dpct --format-range=none -out-root %T/api_recognization %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/api_recognization/api_recognization.dp.cpp
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

namespace at {
    namespace cub {
        void exclusive_scan(){}
    }
}

int main(int argc, char **argv) {
  // CHECK: at::cub::exclusive_scan();
  at::cub::exclusive_scan();

  return 0;

}