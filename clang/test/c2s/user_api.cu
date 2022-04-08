// RUN: c2s --process-all -in-root %S --format-range=none -out-root %T/user_api %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/user_api/user_api.dp.cpp %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: void create_maxpool_cudnn_tensors(){}
// CHECK-NEXT: void foo(){
// CHECK-NEXT:   create_maxpool_cudnn_tensors();
// CHECK-NEXT: }
#include <cuda.h>
void create_maxpool_cudnn_tensors(){}
void foo(){
  create_maxpool_cudnn_tensors();
}

