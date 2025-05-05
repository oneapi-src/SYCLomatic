// RUN: echo "empty command"

#include "cuda_runtime.h"

cudaError_t resetDevice() {
    return cudaDeviceReset();
}
