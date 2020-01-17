// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types002.dp.cpp

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


int main(int argc, char **argv) {
  //TODO: Currently, thrust::device_vector in sizeof is unsupported.
  //CHECK:dpstd::device_vector<int> device_vec;
  //CHECK-NEXT:int a = sizeof(thrust::device_vector<int>);
  //CHECK-NEXT:a = sizeof(device_vec);
  //CHECK-NEXT:a = sizeof device_vec;
  thrust::device_vector<int> device_vec;
  int a = sizeof(thrust::device_vector<int>);
  a = sizeof(device_vec);
  a = sizeof device_vec;

  //TODO: Currently, thrust::device_ptr in sizeof is unsupported.
  //CHECK:dpct::device_ptr<int> device_p;
  //CHECK-NEXT:a = sizeof(thrust::device_ptr<int>);
  //CHECK-NEXT:a = sizeof(device_p);
  //CHECK-NEXT:a = sizeof device_p;
  thrust::device_ptr<int> device_p;
  a = sizeof(thrust::device_ptr<int>);
  a = sizeof(device_p);
  a = sizeof device_p;

  //TODO: Currently, thrust::host_vector in sizeof is unsupported.
  //CHECK:dpstd::host_vector<int> host_vec;
  //CHECK-NEXT:a = sizeof(thrust::host_vector<int>);
  //CHECK-NEXT:a = sizeof(host_vec);
  //CHECK-NEXT:a = sizeof host_vec;
  thrust::host_vector<int> host_vec;
  a = sizeof(thrust::host_vector<int>);
  a = sizeof(host_vec);
  a = sizeof host_vec;
}

