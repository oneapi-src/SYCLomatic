// RUN: dpct --format-range=none -usm-level=none -out-root %T/time-measure-usm-none %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/time-measure-usm-none/time-measure-usm-none.dp.cpp --match-full-lines %s
#include <stdio.h>

#define N 1000

__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}

int main() {
    // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
    // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
    cudaStream_t stream;

    int ha[N], hb[N];
    // CHECK: std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    // CHECK: std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    cudaEvent_t start, stop;
    cudaError_t cudaStatus;

    int *da, *db;
    float elapsedTime;

    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }


    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CHECK: start_ct1 = std::chrono::steady_clock::now();
    cudaEventRecord(start, 0);

    // CHECK: dpct::async_dpct_memcpy(da, ha, N*sizeof(int), dpct::host_to_device);
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    // CHECK: dpct::async_dpct_memcpy(da, ha, N*sizeof(int), dpct::host_to_device);
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, 0);
    // CHECK: dpct::async_dpct_memcpy(da, ha, N*sizeof(int), dpct::host_to_device, *stream);
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, stream);

    // CHECK: q_ct1.wait();
    // CHECK: stream->wait();
    // CHECK: stop_ct1 = std::chrono::steady_clock::now();
    // CHECK: elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    add<<<N, 1>>>(da, db);

    // CHECK: dpct::async_dpct_memcpy(hb, db, N*sizeof(int), dpct::device_to_host);
    cudaMemcpyAsync(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();


    for (int i = 0; i<N; ++i) {
        printf("%d\n", hb[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(da);
    cudaFree(db);

    return 0;
}


__global__ void kernel_foo(){}

void foo_test_1() {

    cudaEvent_t     start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

// CHECK:    start_ct1 = std::chrono::steady_clock::now();
// CHECK-NEXT:        for (int i=0; i<4; i++) {
// CHECK-NEXT:            dpct::get_default_queue().submit(
// CHECK-NEXT:              [&](sycl::handler &cgh) {
// CHECK-NEXT:                cgh.parallel_for<dpct_kernel_name<class kernel_foo_{{[a-z0-9]+}}>>(
// CHECK-NEXT:                  sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:                  [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                    kernel_foo();
// CHECK-NEXT:                  });
// CHECK-NEXT:              });
// CHECK-NEXT:        }
// CHECK-NEXT:    dpct::get_current_device().queues_wait_and_throw();
    cudaEventRecord( start, 0 );
        for (int i=0; i<4; i++) {
            kernel_foo<<<1, 1>>>();
        }
    cudaThreadSynchronize();

    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop ) ;
}
