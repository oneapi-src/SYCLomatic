// RUN: dpct --format-range=none --enable-codepin -out-root %T/debug_test/vector_add_float3 %s --cuda-include-path="%cuda-path/include" -- -std=c++17  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/debug_test/vector_add_float3_codepin_sycl/test.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/debug_test/vector_add_float3_codepin_sycl/test.dp.cpp -o %T/debug_test/vector_add_float3_codepin_sycl/test.dp.o %}
//CHECK: #include <dpct/codepin/codepin.hpp>
//CHECK: #include "codepin_autogen_util.hpp"
#include <iostream>
 
// CUDA kernel: Vector addition for float3
__global__
void vectorAdd(float3* a, float3* b, float3* result, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
 
    // Check if the thread ID is within the vector size
    if (tid < size) {
        result[tid].x = a[tid].x + b[tid].x;
        result[tid].y = a[tid].y + b[tid].y;
        result[tid].z = a[tid].z + b[tid].z;
    }
}
 
int main() {
    const int vectorSize = 10; // Set the size of the vectors
 
    // Host vectors
    float3 *h_a, *h_b, *h_result;
   
    // Allocate memory for host vectors
    h_a = new float3[vectorSize];
    h_b = new float3[vectorSize];
    h_result = new float3[vectorSize];
 
    // Initialize host vectors
    for (int i = 0; i < vectorSize; ++i) {
        h_a[i] = make_float3(1.0f, 2.0f, 3.0f);
        h_b[i] = make_float3(4.0f, 5.0f, 6.0f);
    }
 
    // Device vectors
    float3 *d_a, *d_b, *d_result;
 
    // Allocate memory for device vectors
    //CHECK: dpctexp::codepin::get_ptr_size_map()[*((void**)&d_a)] = vectorSize * sizeof(sycl::float3);
    cudaMalloc((void**)&d_a, vectorSize * sizeof(float3));
    //CHECK: dpctexp::codepin::get_ptr_size_map()[*((void**)&d_b)] = vectorSize * sizeof(sycl::float3);
    cudaMalloc((void**)&d_b, vectorSize * sizeof(float3));
    //CHECK: dpctexp::codepin::get_ptr_size_map()[*((void**)&d_result)] = vectorSize * sizeof(sycl::float3);
    cudaMalloc((void**)&d_result, vectorSize * sizeof(float3));
 
    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, vectorSize * 12, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vectorSize * 12, cudaMemcpyHostToDevice);
 
    // Define grid and block dimensions
    dim3 blockDim(256); // 256 threads per block
    dim3 gridDim((vectorSize + blockDim.x - 1) / blockDim.x); // Sufficient blocks to cover the vector size
 
    // Launch the CUDA kernel
    vectorAdd<<<gridDim, blockDim>>>(d_a, d_b, d_result, vectorSize);
 
    // Copy result from device to host
    cudaMemcpy(h_result, d_result, vectorSize * 12, cudaMemcpyDeviceToHost);
 
    // Print the result
    for (int i = 0; i < 10; ++i) {
        std::cout << "Result[" << i << "]: (" << h_result[i].x << ", " << h_result[i].y << ", " << h_result[i].z << ")\n";
    }
 
    // Free allocated memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
 
    return 0;
}
