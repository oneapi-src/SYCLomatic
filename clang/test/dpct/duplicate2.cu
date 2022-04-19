// RUN: echo duplicate2

__constant__ int ca[32];

// CHECK: void kernel(int *i, int *ca) {
__global__ void kernel(int *i) {
  ca[0] = *i;
}