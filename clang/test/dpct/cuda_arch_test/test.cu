// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T/cuda_arch_test %S/test.cu %s -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %S/test.cu --match-full-lines --input-file %T/cuda_arch_test/test.dp.cpp
// RUN: FileCheck %S/test.h --match-full-lines --input-file %T/cuda_arch_test/test.h
#include "test.h"

// CHECK: class aa{
// CHECK-NEXT: int bb;
// CHECK-NEXT: int aa1(){
// CHECK-NEXT:   #ifdef DPCT_COMPATIBILITY_TEMP
// CHECK-NEXT:     return 1;
// CHECK-NEXT:   #endif
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
// CHECK-NEXT: aa operator+(aa cc){
// CHECK-NEXT:   aa vv;
// CHECK-NEXT:   return vv;
// CHECK-NEXT: }
// CHECK-NEXT: };
class aa{
int bb;
__host__ __device__ int aa1(){
  #ifdef __CUDA_ARCH__
    return 1;
  #endif
  return 0;
}
__host__ __device__ aa operator+(aa cc){
  aa vv;
  return vv;
}
};

// CHECK: static int Env_cuda_thread_in_threadblock(int axis, sycl::nd_item<3> item_ct1)
// CHECK-NEXT: {
// CHECK-NEXT: #ifdef DPCT_COMPATIBILITY_TEMP
// CHECK-NEXT:  return axis==0 ? item_ct1.get_local_id(2) :
// CHECK-NEXT:         axis==1 ? item_ct1.get_local_id(1) :
// CHECK-NEXT:                    item_ct1.get_local_id(0);
// CHECK-NEXT: #else
// CHECK-NEXT:   cudaDeviceSynchronize();
// CHECK-NEXT:   return 0;
// CHECK-NEXT: #endif
// CHECK-NEXT: }
// CHECK-NEXT: static int Env_cuda_thread_in_threadblock_host(int axis)
// CHECK-NEXT: {
// CHECK-NEXT: #ifdef DPCT_NOT_DEFINED
// CHECK-NEXT:   return axis==0 ? threadIdx.x :
// CHECK-NEXT:          axis==1 ? threadIdx.y :
// CHECK-NEXT:                    threadIdx.z;
// CHECK-NEXT: #else
// CHECK-NEXT:   dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:   return 0;
// CHECK-NEXT: #endif
// CHECK-NEXT: }
__host__ __device__ static int Env_cuda_thread_in_threadblock(int axis)
{
#ifdef __CUDA_ARCH__
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#else
  cudaDeviceSynchronize();
  return 0;
#endif
}


// CHECK: template<typename T>
// CHECK-NEXT: int test(T a, T b, sycl::nd_item<3> item_ct1);
// CHECK-NEXT: template<typename T>
// CHECK-NEXT: int test_host(T a, T b);
// CHECK-EMPTY:
// CHECK-NEXT: template<typename T>
// CHECK-NEXT: int test(T a, T b, sycl::nd_item<3> item_ct1){
// CHECK-NEXT: #ifdef DPCT_COMPATIBILITY_TEMP
// CHECK-NEXT:   return item_ct1.get_local_id(2) > 10 ? a : b;
// CHECK-NEXT: #else
// CHECK-NEXT:   cudaDeviceSynchronize();
// CHECK-NEXT:   return a;
// CHECK-NEXT: #endif
// CHECK-NEXT: }
// CHECK-NEXT: template<typename T>
// CHECK-NEXT: int test_host(T a, T b){
// CHECK-NEXT: #ifdef DPCT_NOT_DEFINED
// CHECK-NEXT:   return threadIdx.x > 10 ? a : b;
// CHECK-NEXT: #else
// CHECK-NEXT:   dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:   return a;
// CHECK-NEXT: #endif
// CHECK-NEXT: }
template<typename T>
__host__ __device__ int test(T a, T b);

template<typename T>
__host__ __device__ int test(T a, T b){
#ifdef __CUDA_ARCH__
  return threadIdx.x > 10 ? a : b;
#else
  cudaDeviceSynchronize();
  return a;
#endif
}

// CHECK: int test1(sycl::nd_item<3> item_ct1){
// CHECK-NEXT:   #if DPCT_COMPATIBILITY_TEMP > 800
// CHECK-NEXT:     return threadIdx.x > 8;
// CHECK-NEXT:   #elif DPCT_COMPATIBILITY_TEMP > 700
// CHECK-NEXT:     return threadIdx.x > 7;
// CHECK-NEXT:   #elif DPCT_COMPATIBILITY_TEMP > 600
// CHECK-NEXT:     return threadIdx.x > 6;
// CHECK-NEXT:   #elif DPCT_COMPATIBILITY_TEMP > 500
// CHECK-NEXT:     return item_ct1.get_local_id(2) > 5;
// CHECK-NEXT:   #elif DPCT_COMPATIBILITY_TEMP > 400
// CHECK-NEXT:     return threadIdx.x > 4;
// CHECK-NEXT:   #elif __CUDA_ARCH__ > 300
// CHECK-NEXT:     return threadIdx.x > 3;
// CHECK-NEXT:   #elif __CUDA_ARCH__ > 200
// CHECK-NEXT:     return threadIdx.x > 2;
// CHECK-NEXT:   #elif !defined(__CUDA_ARCH__)
// CHECK-NEXT:     cudaDeviceSynchronize();
// CHECK-NEXT:     return 0;
// CHECK-NEXT:   #endif
// CHECK-NEXT: }
// CHECK-NEXT: int test1_host(){
// CHECK-NEXT:   #if DPCT_NOT_DEFINED > 800
// CHECK-NEXT:     return threadIdx.x > 8;
// CHECK-NEXT:   #elif DPCT_NOT_DEFINED > 700
// CHECK-NEXT:     return threadIdx.x > 7;
// CHECK-NEXT:   #elif DPCT_NOT_DEFINED > 600
// CHECK-NEXT:     return threadIdx.x > 6;
// CHECK-NEXT:   #elif DPCT_NOT_DEFINED > 500
// CHECK-NEXT:     return threadIdx.x > 5;
// CHECK-NEXT:   #elif DPCT_NOT_DEFINED > 400
// CHECK-NEXT:     return threadIdx.x > 4;
// CHECK-NEXT:   #elif DPCT_NOT_DEFINED > 300
// CHECK-NEXT:     return threadIdx.x > 3;
// CHECK-NEXT:   #elif DPCT_NOT_DEFINED > 200
// CHECK-NEXT:     return threadIdx.x > 2;
// CHECK-NEXT:   #elif !defined(DPCT_NOT_DEFINED)
// CHECK-NEXT:     dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:     return 0;
// CHECK-NEXT:   #endif
// CHECK-NEXT: }
__host__ __device__ int test1(){
  #if __CUDA_ARCH__ > 800
    return threadIdx.x > 8;
  #elif __CUDA_ARCH__ > 700
    return threadIdx.x > 7;
  #elif __CUDA_ARCH__ > 600
    return threadIdx.x > 6;
  #elif __CUDA_ARCH__ > 500
    return threadIdx.x > 5;
  #elif __CUDA_ARCH__ > 400
    return threadIdx.x > 4;
  #elif __CUDA_ARCH__ > 300
    return threadIdx.x > 3;
  #elif __CUDA_ARCH__ > 200
    return threadIdx.x > 2;
  #elif !defined(__CUDA_ARCH__)
    cudaDeviceSynchronize();
    return 0;
  #endif
}

// CHECK: void kernel(sycl::nd_item<3> item_ct1){
// CHECK-NEXT:   float a, b;
// CHECK-NEXT:   Env_cuda_thread_in_threadblock(0, item_ct1);
// CHECK-NEXT:   test(0, 0, item_ct1);
// CHECK-NEXT:   test<float>(a, b, item_ct1);
// CHECK-NEXT:   test1(item_ct1);
// CHECK-NEXT: }
__global__ void kernel(){
  float a, b;
  Env_cuda_thread_in_threadblock(0);
  test(0, 0);
  test<float>(a, b);
  test1();
}

int main(){
float a, b;
// CHECK: test_host(a, b);
// CHECK-NEXT: test_host<int>(1, 1);
// CHECK-NEXT: Env_cuda_thread_in_threadblock_host(0);
// CHECK-NEXT: test1_host();
test(a, b);
test<int>(1, 1);
Env_cuda_thread_in_threadblock(0);
test1();
kernel<<<1,1>>>();
cudaDeviceSynchronize();

return 0;
}
