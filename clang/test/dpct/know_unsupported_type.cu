// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.2
// RUN: dpct -out-root %T/know_unsupported_type %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/know_unsupported_type/know_unsupported_type.dp.cpp
#include "cuda_runtime.h"
#include "cuda.h"
#include "cusparse.h"
#include "nvml.h"
#include <vector>
int main(int argc, char **argv) {
    // CHECK: dpct::image_matrix_desc *pcad;
    CUDA_ARRAY_DESCRIPTOR *pcad;
    // CHECK: dpct::memcpy_parameter *p1c3d;
    cudaMemcpy3DParms *p1c3d;
    // CHECK: const dpct::memcpy_parameter *p2c3d;
    const cudaMemcpy3DParms *p2c3d;
    // CHECK: static dpct::memcpy_parameter *p3c3d;
    static cudaMemcpy3DParms *p3c3d;
    // CHECK: static volatile dpct::memcpy_parameter *p4c3d;
    static volatile cudaMemcpy3DParms *p4c3d;
    // CHECK: std::vector<dpct::memcpy_parameter *> vc3dp;
    std::vector<cudaMemcpy3DParms *> vc3dp;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of CUexternalMemory type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUexternalMemory cum;
    CUexternalMemory cum;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of CUexternalSemaphore type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUexternalSemaphore cus;
    CUexternalSemaphore cus;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of CUgraph type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUgraph cug;
    CUgraph cug;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of CUgraphExec type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUgraphExec cuge;
    CUgraphExec cuge;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of CUgraphNode type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUgraphNode cugn;
    CUgraphNode cugn;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of nvmlDevice_t type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: nvmlDevice_t nvmld;
    nvmlDevice_t nvmld;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of nvmlReturn_t type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: nvmlReturn_t nvmlr;
    nvmlReturn_t nvmlr;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of nvmlMemory_t type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: nvmlMemory_t nvmlm;
    nvmlMemory_t nvmlm;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of nvmlValueType_t type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: nvmlValueType_t nvmlvt;
    nvmlValueType_t nvmlvt;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: Migration of nvmlValue_t type is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: nvmlValue_t nvmlv;
    nvmlValue_t nvmlv;

    return 0;
}

