#include "cuda_runtime.h"

cudaError_t resetDevice() {
    return cudaDeviceReset();
}
