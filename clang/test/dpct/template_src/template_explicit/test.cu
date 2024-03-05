// RUN: echo "Empty command."
#include "test.hpp"

// A host function to launch the CUDA kernel
void runNbnxnPruneKernel(int numParts, bool freshList) {
    if (freshList) {
        nbnxn_kernel_prune_cuda<true><<<1, 1>>>(numParts);
    } else {
        nbnxn_kernel_prune_cuda<false><<<1, 1>>>(numParts);
    }
}

