// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/malloc.dp.cpp
#include <cuda_runtime.h>

template <class T>
void runTest(int len);

template<class T>
void runTest(int len){
    T *d_idata;
    // CHECK: d_idata = (T *)cl::sycl::malloc_device(sizeof(T), dpct::get_current_device(), dpct::get_default_context());
    cudaMalloc((void **) &d_idata, sizeof(T));
}

typedef struct {
    float x;
    float y;
    int cluster;
} Point;

int main(){
    runTest<float2>(32);
    runTest<int2>(64);


    int2 * d_data_int2;
    // CHECK: d_data_int2 = (cl::sycl::int2 *)cl::sycl::malloc_device(sizeof(cl::sycl::int2), dpct::get_current_device(), dpct::get_default_context());
    cudaMalloc((void **) &d_data_int2, sizeof(int2));

    Point * d_data_Point;
    // CHECK: d_data_Point = (Point *)cl::sycl::malloc_device(sizeof(Point), dpct::get_current_device(), dpct::get_default_context());
    cudaMalloc((void **) &d_data_Point, sizeof(Point));

    const int2 * d_const_int2;
    // CHECK: d_const_int2 = (cl::sycl::int2 const *)cl::sycl::malloc_device(sizeof(cl::sycl::int2), dpct::get_current_device(), dpct::get_default_context());
    cudaMalloc((void **) &d_const_int2, sizeof(int2));

    int2 const * volatile * d_data;
    // CHECK: d_data = (cl::sycl::int2 const * volatile *)cl::sycl::malloc_device(sizeof(cl::sycl::int2), dpct::get_current_device(), dpct::get_default_context());
    cudaMalloc((void **) &d_data, sizeof(int2));
}
