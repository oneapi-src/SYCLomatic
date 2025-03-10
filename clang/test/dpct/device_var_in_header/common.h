//CHECK: inline dpct::constant_memory<int, 1> arr(sycl::range<1>(2), {1, 2});
__device__ __constant__ int arr[2] = {1, 2};
//CHECK: static dpct::constant_memory<int, 1> arr1(sycl::range<1>(2), {1, 2});
static __device__ __constant__ int arr1[2] = {1, 2};

__global__ void f();
