// RUN: cd %T
// RUN: cat %S/compile_commands.json > %T/compile_commands.json
// RUN: cat %S/test.c > %T/test.c

// RUN: dpct  -p=%T  -in-root=%T -out-root=%T/out -format-range=none --cuda-include-path="%cuda-path/include"

// RUN: FileCheck  %s --match-full-lines --input-file %T/out/test.c.dp.cpp

#include <cuda.h>
#include <cuda_runtime.h>

enum test_enum {
	test_1
};
int main() {

    cudaError_t cudares = cudaFree(0);
    // CHECK: if(cudares != 0) return(-1);
    if(cudares != cudaSuccess) return(-1);

    CUevent event;
    static CUcontext ctx;
    // CHECK: if (0) {
    if (CUDA_SUCCESS) {
        return 0;
    }
    // CHECK: dpct::device_info deviceProp;
    struct cudaDeviceProp deviceProp;

}
