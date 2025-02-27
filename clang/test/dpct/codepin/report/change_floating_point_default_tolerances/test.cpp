// UNSUPPORTED: system-windows
// RUN: cat %S/cuda.json > %T/cuda.json
// RUN: cat %S/cuda.bin > %T/cuda.bin
// RUN: cat %S/sycl.json > %T/sycl.json
// RUN: cat %S/sycl.bin > %T/sycl.bin
// RUN: cat %S/tolerances.json> %T/tolerances.json
// RUN: cd %T
// RUN: dpct --codepin-report --instrumented-cuda-log cuda.json --instrumented-sycl-log sycl.json --floating-point-comparison-epsilon=tolerances.json  || true

// RUN: cat %S/CodePin_Report_ref.csv > %T/CodePin_Report_Expected.csv
// RUN: cat %T/CodePin_Report.csv >> %T/CodePin_Report_Expected.csv

// RUN: FileCheck --match-full-lines --input-file %T/CodePin_Report_Expected.csv %T/CodePin_Report_Expected.csv


#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    cudaError_t err = (func);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error calling \"" #func "\", error code is " << err   \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  }

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 5;
    size_t size = numElements * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(i);  // # change the float to i + 0.00001 and co-work with the tolerance file to compare the float value.
        h_B[i] = static_cast<float>(i * 2);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CALL(cudaMalloc((void**)&d_A, size));
    CUDA_CALL(cudaMalloc((void**)&d_B, size));
    CUDA_CALL(cudaMalloc((void**)&d_C, size));

    CUDA_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            exit(1);
        }
    }

    std::cout << "Test PASSED" << std::endl;

    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}