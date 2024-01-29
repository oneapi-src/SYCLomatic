// RUN: echo "test"
// SYCL: #include <sycl/sycl.hpp>
// SYCL-NEXT: #include <dpct/dpct.hpp>
// SYCL: void foo() {
// SYCL-NEXT:   sycl::float2 f2;
// SYCL-NEXT: }
// CUDA: void foo() {
// CUDA-NEXT:   float2 f2;
// CUDA-NEXT: }
void foo() {
  float2 f2;
}
