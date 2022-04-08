// RUN: c2s --format-range=none -out-root %T/malloc %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/malloc/malloc.dp.cpp
#include <cuda_runtime.h>

template <class T>
void runTest(int len);

template<class T>
void runTest(int len){
    T *d_idata;
    // CHECK: d_idata = (T *)sycl::malloc_device(sizeof(T), c2s::get_default_queue());
    cudaMalloc((void **) &d_idata, sizeof(T));
}

typedef struct {
    float x;
    float y;
    int cluster;
} Point;

int main(){
    //CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
    //CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
    runTest<float2>(32);
    runTest<int2>(64);


    int2 * d_data_int2;
    // CHECK: d_data_int2 = sycl::malloc_device<sycl::int2>(1, q_ct1);
    cudaMalloc((void **) &d_data_int2, sizeof(int2));

    Point * d_data_Point;
    // CHECK: d_data_Point = sycl::malloc_device<Point>(1, q_ct1);
    cudaMalloc((void **) &d_data_Point, sizeof(Point));

    const int2 * d_const_int2;
    // CHECK: d_const_int2 = (const sycl::int2 *)sycl::malloc_device(sizeof(sycl::int2), q_ct1);
    cudaMalloc((void **) &d_const_int2, sizeof(int2));

    int2 const * volatile * d_data;
    // CHECK: d_data = (const sycl::int2 *volatile *)sycl::malloc_device(sizeof(sycl::int2), q_ct1);
    cudaMalloc((void **) &d_data, sizeof(int2));


#define INT2 int2
    // CHECK: d_data_int2 = (sycl::int2 *)sycl::malloc_device(sizeof(INT2)*100, q_ct1);
    cudaMalloc(&d_data_int2, sizeof(INT2)*100);
    // CHECK: d_data_int2 = (sycl::int2 *)sycl::malloc_device(sizeof(INT2)*100, q_ct1);
    cudaMalloc((void **)&d_data_int2, sizeof(INT2)*100);
}

