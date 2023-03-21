// UNSUPPORTED: system-windows
// RUN: echo "empty command"

#include <iostream>
#include <cuda_runtime.h>
int main() {
    dim3 blockSize(256);
    return 0;
}
