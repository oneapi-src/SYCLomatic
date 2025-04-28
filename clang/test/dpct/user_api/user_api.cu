// RUN: dpct --in-root %S --format-range=none --out-root %T %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/user_api.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/user_api.dp.cpp -o %T/user_api.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda.h>

// CHECK: void create_maxpool_cudnn_tensors() {}
// CHECK-NEXT: void foo1() {
// CHECK-NEXT:   create_maxpool_cudnn_tensors();
// CHECK-NEXT: }
void create_maxpool_cudnn_tensors() {}
void foo1() {
  create_maxpool_cudnn_tensors();
}

void tex2D(int a, int b) {}
void tex2D(int a, int b, int c) {}
void tex3D(int a, int b, int c) {}

// CHECK: void foo2() {
// CHECK-NEXT:   tex2D(0, 0);
// CHECK-NEXT:   tex2D(0, 0, 0);
// CHECK-NEXT:   tex3D(0, 0, 0);
// CHECK-NEXT: }
void foo2() {
  tex2D(0, 0);
  tex2D(0, 0, 0);
  tex3D(0, 0, 0);
}
