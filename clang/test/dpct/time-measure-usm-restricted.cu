// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/time-measure-usm-restricted.dp.cpp --match-full-lines %s
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

    // CHECK: sycl::event stop_q_ct1_1 = q_ct1.memcpy(da, ha, N*sizeof(int));
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    // CHECK: sycl::event stop_q_ct1_2 = q_ct1.memcpy(da, ha, N*sizeof(int));
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, 0);
    // CHECK: sycl::event stop_stream_1 = stream->memcpy(da, ha, N*sizeof(int));
    cudaMemcpyAsync(da, ha, N*sizeof(int), cudaMemcpyHostToDevice, stream);

    // CHECK: stop_q_ct1_1.wait();
    // CHECK: stop_q_ct1_2.wait();
    // CHECK: stop_stream_1.wait();
    // CHECK: stop_ct1 = std::chrono::steady_clock::now();
    // CHECK: elapsedTime = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    add<<<N, 1>>>(da, db);

    // CHECK: q_ct1.memcpy(hb, db, N*sizeof(int));
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

