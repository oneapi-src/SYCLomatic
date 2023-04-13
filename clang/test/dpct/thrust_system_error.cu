// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.2, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/thrust_system_error %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust_system_error/thrust_system_error.dp.cpp --match-full-lines %s

#include <cuda_runtime_api.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <string>

void cuda_safe_call(cudaError_t error, const std::string& message = "")
{
  if(error) {
// CHECK: throw std::system_error(error, std::generic_category(), message);
    throw thrust::system_error(error, thrust::cuda_category(), message);
  }
}

int main() {
// CHECK: dpct::err0 e = 1;  
  cudaError_t e = cudaErrorInvalidValue;  
  cuda_safe_call(e);
  return 0;
} 
