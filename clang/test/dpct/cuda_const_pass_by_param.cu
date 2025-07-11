
// RUN: dpct --format-range=none --usm-level=none -out-root %T/cuda_const_pass_by_param %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_const_pass_by_param/cuda_const_pass_by_param.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda_const_pass_by_param/cuda_const_pass_by_param.dp.cpp -o %T/cuda_const_pass_by_param/cuda_const_pass_by_param.dp.o %}
#include <cstdio>
#include <cuda_runtime.h>

#define MAX_CONST_SIZE 1024
__constant__ char device_const_buffer[MAX_CONST_SIZE];


__host__  void* qudaGetSymbolAddress(const void* symbol) {

    void* ptr;
    // CHECK: *(&ptr) = const_cast<void *>(symbol);
    cudaGetSymbolAddress(&ptr, symbol); 
    return ptr;

}

__host__  void* qudaGetSymbolAddress2() {

    void* ptr;
    // CHECK:  *(&ptr) = device_const_buffer.get_ptr();
    cudaGetSymbolAddress(&ptr, device_const_buffer);
    return ptr;

}


template <typename T>
__host__ void process_buffer(T* data) {
    
    if(data) printf("Processed: %f\n", static_cast<float>(data[0]));
}


int main() {
    float h_data[256];
    for(int i=0; i<256; i++) h_data[i] = i*1.0f;
// CHECK: dpct::dpct_memcpy(device_const_buffer.get_ptr(), h_data, sizeof(h_data));
    cudaMemcpyToSymbol(device_const_buffer, h_data, sizeof(h_data));
// CHECK: void* host_ptr = qudaGetSymbolAddress(device_const_buffer.get_ptr());
    void* host_ptr = qudaGetSymbolAddress(device_const_buffer);
    void* host_ptr2 = qudaGetSymbolAddress2();
    process_buffer<float>(static_cast<float*>(host_ptr));
    cudaDeviceSynchronize();
    
    return 0;
}


