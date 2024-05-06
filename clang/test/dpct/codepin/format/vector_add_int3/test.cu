// RUN: dpct --format-range=none --enable-codepin -out-root %T/out %s --cuda-include-path="%cuda-path/include" -- -std=c++17  -x cuda --cuda-host-only

// RUN: cd %T
// RUN: %if build_lit %{icpx -fsycl  %T/out_codepin_sycl/test.dp.cpp -o %T/test.run %}
// RUN: %if build_lit %{ ./test.run %}
// RUN: %if build_lit %{ cat %S/app_runtime_data_record.json_ref >> %T/app_runtime_data_record.json_ref %}
// RUN: %if build_lit %{ cat %T/CodePin_SYCL_*.json >> %T/app_runtime_data_record.json_ref %}
// RUN: %if build_lit %{ FileCheck --match-full-lines --input-file %T/app_runtime_data_record.json_ref %T/app_runtime_data_record.json_ref %}
#include <iostream>
 
// CUDA kernel: Vector addition for int3
__global__
void vectorAdd(int3* a, int3* b, int3* result, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
 
    // Check if the thread ID is within the vector size
    if (tid < size) {
        result[tid].x = a[tid].x + b[tid].x;
        result[tid].y = a[tid].y + b[tid].y;
        result[tid].z = a[tid].z + b[tid].z;
    }
}
 
int main() {
    const int vectorSize = 1; // Set the size of the vectors
 
    // Host vectors
    int3 *h_a, *h_b, *h_result;
   
    // Allocate memory for host vectors
    h_a = new int3[vectorSize];
    h_b = new int3[vectorSize];
    h_result = new int3[vectorSize];
 
    // Initialize host vectors
    for (int i = 0; i < vectorSize; ++i) {
        h_a[i] = make_int3(1, 2, 3);
        h_b[i] = make_int3(4, 5, 6);
    }
 
    // Device vectors
    int3 *d_a, *d_b, *d_result;
 
    // Allocate memory for device vectors
    //CHECK: dpctexp::codepin::get_ptr_size_map()[*((void**)&d_a)] = vectorSize * sizeof(sycl::int3);
    cudaMalloc((void**)&d_a, vectorSize * sizeof(int3));
    //CHECK: dpctexp::codepin::get_ptr_size_map()[*((void**)&d_b)] = vectorSize * sizeof(sycl::int3);
    cudaMalloc((void**)&d_b, vectorSize * sizeof(int3));
    //CHECK: dpctexp::codepin::get_ptr_size_map()[*((void**)&d_result)] = vectorSize * sizeof(sycl::int3);
    cudaMalloc((void**)&d_result, vectorSize * sizeof(int3));
 
    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, vectorSize * 12, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vectorSize * 12, cudaMemcpyHostToDevice);
 
    // Define grid and block dimensions
    dim3 blockDim(256); // 256 threads per block
    dim3 gridDim((vectorSize + blockDim.x - 1) / blockDim.x); // Sufficient blocks to cover the vector size
 
    // Launch the CUDA kernel
    //CHECK: dpctexp::codepin::gen_prolog_API_CP("{{[._0-9a-zA-Z\/\(\)\:]+}}", &q_ct1, "d_a", d_a, "d_b", d_b, "d_result", d_result, "vectorSize", vectorSize);
    vectorAdd<<<gridDim, blockDim, 0, 0>>>(d_a, d_b, d_result, vectorSize);
    //CHECK: dpctexp::codepin::gen_epilog_API_CP("{{[._0-9a-zA-Z\/\(\)\:]+}}", &q_ct1, "d_a", d_a, "d_b", d_b, "d_result", d_result, "vectorSize", vectorSize);
 
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
