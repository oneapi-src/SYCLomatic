// RUN: dpct --format-range=none --usm-level=none -in-root %S -out-root %T/cuda_arch_test %S/test.cu -extra-arg="-I %S" --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
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

// CHECK: static int Env_cuda_thread_in_threadblock(int axis,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1)
// CHECK-NEXT: {
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   return axis==0 ? item_ct1.get_local_id(2) :
// CHECK-NEXT:          axis==1 ? item_ct1.get_local_id(1) :
// CHECK-NEXT:                    item_ct1.get_local_id(0);
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT:   return axis==0 ? item_ct1.get_local_id(2) :
// CHECK-NEXT:          axis==1 ? item_ct1.get_local_id(1) :
// CHECK-NEXT:                    item_ct1.get_local_id(0);
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT:   return axis==0 ? item_ct1.get_local_id(2) :
// CHECK-NEXT:          axis==1 ? item_ct1.get_local_id(1) :
// CHECK-NEXT:                    item_ct1.get_local_id(0);
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
// CHECK-NEXT: static int Env_cuda_thread_in_threadblock_host_ct{{[0-9]+}}(int axis)
// CHECK-NEXT: {
// CHECK-NEXT: dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   dev_ct1.queues_wait_and_throw();
// CHECK-NEXT:   return 0;
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT:   dev_ct1.queues_wait_and_throw();
// CHECK-NEXT:   return 0;
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT:   dev_ct1.queues_wait_and_throw();
// CHECK-NEXT:   return 0;
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
__host__ __device__ static int Env_cuda_thread_in_threadblock(int axis)
{
  int a = 1;
#ifdef __CUDA_ARCH__
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#else
  cudaDeviceSynchronize();
  return 0;
#endif

#if !defined(__CUDA_ARCH__)
  cudaDeviceSynchronize();
  return 0;
#else
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#endif

#if defined(__CUDA_ARCH__)
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#else
  cudaDeviceSynchronize();
  return 0;
#endif

  return a;
}

// CHECK: static int Env_cuda_thread_in_threadblock1(int axis,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1)
// CHECK-NEXT: {
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   return axis==0 ? item_ct1.get_local_id(2) :
// CHECK-NEXT:          axis==1 ? item_ct1.get_local_id(1) :
// CHECK-NEXT:                    item_ct1.get_local_id(0);
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
// CHECK-NEXT: static int Env_cuda_thread_in_threadblock1_host_ct{{[0-9]+}}(int axis)
// CHECK-NEXT: {
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:   return 0;
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
__host__ __device__ static int Env_cuda_thread_in_threadblock1(int axis)
{
  int a = 1;
#if defined(__CUDA_ARCH__)
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#else
  cudaDeviceSynchronize();
  return 0;
#endif
  return a;
}

// CHECK: static int Env_cuda_thread_in_threadblock2(int axis,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1)
// CHECK-NEXT: {
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   return axis==0 ? item_ct1.get_local_id(2) :
// CHECK-NEXT:          axis==1 ? item_ct1.get_local_id(1) :
// CHECK-NEXT:                    item_ct1.get_local_id(0);
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
// CHECK-NEXT: static int Env_cuda_thread_in_threadblock2_host_ct{{[0-9]+}}(int axis)
// CHECK-NEXT: {
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:   return 0;
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
__host__ __device__ static int Env_cuda_thread_in_threadblock2(int axis)
{
  int a = 1;
#if __CUDA_ARCH__
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#else
  cudaDeviceSynchronize();
  return 0;
#endif
  return a;
}

// CHECK: static int Env_cuda_thread_in_threadblock3(int axis,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1)
// CHECK-NEXT: {
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   return axis==0 ? item_ct1.get_local_id(2) :
// CHECK-NEXT:          axis==1 ? item_ct1.get_local_id(1) :
// CHECK-NEXT:                    item_ct1.get_local_id(0);
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
// CHECK-NEXT: static int Env_cuda_thread_in_threadblock3_host_ct{{[0-9]+}}(int axis)
// CHECK-NEXT: {
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:   return 0;
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
__host__ __device__ static int Env_cuda_thread_in_threadblock3(int axis)
{
  int a = 1;
#ifndef __CUDA_ARCH__
  cudaDeviceSynchronize();
  return 0;
#else
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#endif
  return a;
}

// CHECK: static int Env_cuda_thread_in_threadblock4(int axis,
// CHECK-NEXT: const sycl::nd_item<3> &item_ct1)
// CHECK-NEXT: {
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   return axis==0 ? item_ct1.get_local_id(2) :
// CHECK-NEXT:          axis==1 ? item_ct1.get_local_id(1) :
// CHECK-NEXT:                    item_ct1.get_local_id(0);
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
// CHECK-NEXT: static int Env_cuda_thread_in_threadblock4_host_ct{{[0-9]+}}(int axis)
// CHECK-NEXT: {
// CHECK-NEXT:   int a = 1;
// CHECK-EMPTY:
// CHECK-NEXT:   dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:   return 0;
// CHECK-EMPTY:
// CHECK-NEXT:   return a;
// CHECK-NEXT: }
__host__ __device__ static int Env_cuda_thread_in_threadblock4(int axis)
{
  int a = 1;
#if !defined(__CUDA_ARCH__)
  cudaDeviceSynchronize();
  return 0;
#else
  return axis==0 ? threadIdx.x :
         axis==1 ? threadIdx.y :
                   threadIdx.z;
#endif
  return a;
}

// CHECK: template<typename T>
// CHECK-NEXT: int test(T a, T b, const sycl::nd_item<3> &item_ct1);
// CHECK-NEXT: template<typename T>
// CHECK-NEXT: int test_host_ct{{[0-9]+}}(T a, T b);
// CHECK-EMPTY:
// CHECK-NEXT: template<typename T>
// CHECK-NEXT: int test(T a, T b, const sycl::nd_item<3> &item_ct1){
// CHECK-EMPTY:
// CHECK-NEXT:   return item_ct1.get_local_id(2) > 10 ? a : b;
// CHECK-EMPTY:
// CHECK-NEXT: }
// CHECK-NEXT: template<typename T>
// CHECK-NEXT: int test_host_ct{{[0-9]+}}(T a, T b){
// CHECK-EMPTY:
// CHECK-NEXT:   dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:   return a;
// CHECK-EMPTY:
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

// CHECK: int test1(const sycl::nd_item<3> &item_ct1){
// CHECK-NEXT:   #if DPCT_COMPATIBILITY_TEMP > 800
// CHECK-NEXT:     return item_ct1.get_local_id(2) > 8;
// CHECK-NEXT:   #elif DPCT_COMPATIBILITY_TEMP > 700
// CHECK-NEXT:     return threadIdx.x > 7;
// CHECK-NEXT:   #elif __CUDA_ARCH__ > 600
// CHECK-NEXT:     return threadIdx.x > 6;
// CHECK-NEXT:   #elif __CUDA_ARCH__ > 500
// CHECK-NEXT:     return threadIdx.x > 5;
// CHECK-NEXT:   #elif __CUDA_ARCH__ > 400
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
// CHECK-NEXT: int test1_host_ct{{[0-9]+}}(){
// CHECK-NEXT:   #if !DPCT_COMPATIBILITY_TEMP > 800
// CHECK-NEXT:     return threadIdx.x > 8;
// CHECK-NEXT:   #elif !DPCT_COMPATIBILITY_TEMP > 700
// CHECK-NEXT:     return threadIdx.x > 7;
// CHECK-NEXT:   #elif !DPCT_COMPATIBILITY_TEMP > 600
// CHECK-NEXT:     return threadIdx.x > 6;
// CHECK-NEXT:   #elif !DPCT_COMPATIBILITY_TEMP > 500
// CHECK-NEXT:     return threadIdx.x > 5;
// CHECK-NEXT:   #elif !DPCT_COMPATIBILITY_TEMP > 400
// CHECK-NEXT:     return threadIdx.x > 4;
// CHECK-NEXT:   #elif !DPCT_COMPATIBILITY_TEMP > 300
// CHECK-NEXT:     return threadIdx.x > 3;
// CHECK-NEXT:   #elif !DPCT_COMPATIBILITY_TEMP > 200
// CHECK-NEXT:     return threadIdx.x > 2;
// CHECK-NEXT:   #elif defined(DPCT_COMPATIBILITY_TEMP)
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

// CHECK: int test2(){
// CHECK-NEXT:   #ifdef TEST_MACRO
// CHECK-NEXT:     return 0;
// CHECK-NEXT:   #else
// CHECK-NEXT:     return 1;
// CHECK-NEXT:   #endif
// CHECK-NEXT: }
__host__ __device__ int test2(){
  #ifdef TEST_MACRO
    return 0;
  #else
    return 1;
  #endif
}

// CHECK: void kernel(const sycl::nd_item<3> &item_ct1){
// CHECK-NEXT:   float a, b;
// CHECK-NEXT:   Env_cuda_thread_in_threadblock(0, item_ct1);
// CHECK-NEXT:   Env_cuda_thread_in_threadblock1(0, item_ct1);
// CHECK-NEXT:   Env_cuda_thread_in_threadblock2(0, item_ct1);
// CHECK-NEXT:   Env_cuda_thread_in_threadblock3(0, item_ct1);
// CHECK-NEXT:   Env_cuda_thread_in_threadblock4(0, item_ct1);
// CHECK-NEXT:   test(0, 0, item_ct1);
// CHECK-NEXT:   test<float>(a, b, item_ct1);
// CHECK-NEXT:   test1(item_ct1);
// CHECK-NEXT:   test2();
// CHECK-NEXT: }
__global__ void kernel(){
  float a, b;
  Env_cuda_thread_in_threadblock(0);
  Env_cuda_thread_in_threadblock1(0);
  Env_cuda_thread_in_threadblock2(0);
  Env_cuda_thread_in_threadblock3(0);
  Env_cuda_thread_in_threadblock4(0);
  test(0, 0);
  test<float>(a, b);
  test1();
  test2();
}

// CHECK:  int test3(){
// CHECK:    return 1;
// CHECK:  }
// CHECK:  int test3_host_ct{{[0-9]+}}(){
// CHECK:    return 0;
// CHECK:  }
__host__ __device__ int test3(){
#ifdef __CUDA_ARCH__
  return 1;
#else
  return 0;
#endif
}

// CHECK:  int test4(){
// CHECK:    return test3();
// CHECK:  }
// CHECK:  int test4_host_ct{{[0-9]+}}(){
// CHECK:    return test3_host_ct{{[0-9]+}}();
// CHECK:  }
__host__ __device__ int test4(){
#ifdef __CUDA_ARCH__
  return test3();
#else
  return test3();
#endif
}

// CHECK:  int test5(){
// CHECK:    return test4();
// CHECK:  }
// CHECK:  int test5_host_ct{{[0-9]+}}(){
// CHECK:    return test4_host_ct{{[0-9]+}}();
// CHECK:  }
__host__ __device__ int test5(){
  #ifdef __CUDA_ARCH__
    return test4();
  #else
    return test4();
  #endif
} 

int main(){
float a, b;
// CHECK: test_host_ct{{[0-9]+}}(a, b);
// CHECK-NEXT: test_host_ct{{[0-9]+}}<int>(1, 1);
// CHECK-NEXT: Env_cuda_thread_in_threadblock_host_ct{{[0-9]+}}(0);
// CHECK-NEXT: Env_cuda_thread_in_threadblock1_host_ct{{[0-9]+}}(0);
// CHECK-NEXT: Env_cuda_thread_in_threadblock2_host_ct{{[0-9]+}}(0);
// CHECK-NEXT: Env_cuda_thread_in_threadblock3_host_ct{{[0-9]+}}(0);
// CHECK-NEXT: Env_cuda_thread_in_threadblock4_host_ct{{[0-9]+}}(0);
// CHECK-NEXT: test1_host_ct{{[0-9]+}}();
// CHECK-NEXT: test2();
// CHECK-NEXT: test5_host_ct{{[0-9]+}}();
test(a, b);
test<int>(1, 1);
Env_cuda_thread_in_threadblock(0);
Env_cuda_thread_in_threadblock1(0);
Env_cuda_thread_in_threadblock2(0);
Env_cuda_thread_in_threadblock3(0);
Env_cuda_thread_in_threadblock4(0);
test1();
test2();
test5();
kernel<<<1,1>>>();
cudaDeviceSynchronize();

return 0;
}
