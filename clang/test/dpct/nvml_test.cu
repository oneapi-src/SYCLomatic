// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.2
// RUN: dpct -out-root %T/nvml_test %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/nvml_test/nvml_test.dp.cpp
#include "cuda_runtime.h"
#include "nvml.h"
#include <vector>

int main() {
// CHECK: /*
// CHECK: DPCT1007:0: Migration of nvmlInit_v2 is not supported.
// CHECK: */
// CHECK: nvmlInit();
    nvmlInit();

// CHECK: /*
// CHECK: DPCT1007:1: Migration of nvmlInit_v2 is not supported.
// CHECK: */
// CHECK: nvmlInit_v2();
    nvmlInit_v2();

    char Ver[10];
// CHECK: /*
// CHECK: DPCT1007:2: Migration of nvmlSystemGetDriverVersion is not supported.
// CHECK: */
// CHECK: nvmlSystemGetDriverVersion(Ver, 10);
    nvmlSystemGetDriverVersion(Ver, 10);

    unsigned int dc;
// CHECK: /*
// CHECK: DPCT1007:3: Migration of nvmlDeviceGetCount_v2 is not supported.
// CHECK: */
// CHECK: nvmlDeviceGetCount_v2(&dc);
    nvmlDeviceGetCount_v2(&dc);

// CHECK: /*
// CHECK: DPCT1007:4: Migration of nvmlShutdown is not supported.
// CHECK: */
// CHECK: nvmlShutdown();
    nvmlShutdown();

    return 0;
}