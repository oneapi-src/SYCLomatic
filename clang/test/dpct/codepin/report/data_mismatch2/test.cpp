// UNSUPPORTED: system-windows
// RUN: cat %S/cuda.json > %T/cuda.json
// RUN: cat %S/sycl.json > %T/sycl.json
// RUN: cd %T
// RUN: codepin-report.py --instrumented-cuda-log cuda.json --instrumented-sycl-log sycl.json || true

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

int main() {
    float host_data[10];
    for (int i = 0; i < 10; ++i) {
        host_data[i] = static_cast<float>(i);
    }

    float* device_data;
    CUDA_CALL(cudaMalloc((void**)&device_data, 10 * sizeof(float)));

    CUDA_CALL(cudaMemcpy(device_data, host_data, 10 * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;

    CUDA_CALL(cudaFree(device_data));
        // dpctexp::codepin::gen_epilog_API_CP(
        // "test_gen.cu:40:5",
        // &dpct::get_in_order_queue(), "device_data", device_data);
    return 0;
}