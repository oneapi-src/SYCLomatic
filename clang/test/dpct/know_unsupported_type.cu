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
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type CUDA_ARRAY_DESCRIPTOR * was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUDA_ARRAY_DESCRIPTOR *pcad;
    CUDA_ARRAY_DESCRIPTOR *pcad;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type cudaMemcpy3DParms * was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: cudaMemcpy3DParms *p1c3d;
    cudaMemcpy3DParms *p1c3d;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type cudaMemcpy3DParms * was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: const cudaMemcpy3DParms *p2c3d;
    const cudaMemcpy3DParms *p2c3d;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type cudaMemcpy3DParms * was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: static cudaMemcpy3DParms *p3c3d;
    static cudaMemcpy3DParms *p3c3d;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type cudaMemcpy3DParms * was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: static volatile cudaMemcpy3DParms *p4c3d;
    static volatile cudaMemcpy3DParms *p4c3d;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type cudaMemcpy3DParms * was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: std::vector<cudaMemcpy3DParms *> vc3dp;
    std::vector<cudaMemcpy3DParms *> vc3dp;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type cusparseSolvePolicy_t was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: std::vector<cusparseSolvePolicy_t> vsp;
    std::vector<cusparseSolvePolicy_t> vsp;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type cusparseSolvePolicy_t was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: volatile cusparseSolvePolicy_t SP;
    volatile cusparseSolvePolicy_t SP;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type CUexternalMemory was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUexternalMemory cum;
    CUexternalMemory cum;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type CUexternalSemaphore was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUexternalSemaphore cus;
    CUexternalSemaphore cus;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type CUgraph was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUgraph cug;
    CUgraph cug;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type CUgraphExec was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUgraphExec cuge;
    CUgraphExec cuge;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type CUgraphNode was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUgraphNode cugn;
    CUgraphNode cugn;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type CUgraphicsResource was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUgraphicsResource cugr;
    CUgraphicsResource cugr;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type nvmlDevice_t was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: nvmlDevice_t nvmld;
    nvmlDevice_t nvmld;
    // CHECK: /*
    // CHECK-NEXT: DPCT1082:{{[0-9]+}}: The type nvmlReturn_t was not migrated, because the Intel(R) DPC++ Compatibility Tool doesn't support migration of this type.
    // CHECK-NEXT: */
    // CHECK-NEXT: nvmlReturn_t nvmlr;
    nvmlReturn_t nvmlr;

    return 0;
}

