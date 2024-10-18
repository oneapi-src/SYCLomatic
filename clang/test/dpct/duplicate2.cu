// RUN: echo duplicate2
#ifndef  NO_BUILD_TEST
__constant__ int ca[32];

// CHECK: void kernel(int *i, int const *ca) {
__global__ void kernel(int *i) {
  ca[0] = *i;
}
#endif
