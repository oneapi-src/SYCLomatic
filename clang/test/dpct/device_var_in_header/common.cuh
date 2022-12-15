//CHECK: inline dpct::constant_memory<int, 1> arr2(sycl::range<1>(2), {1, 2});
__device__ __constant__ int arr2[2] = {1, 2};
//CHECK: static dpct::constant_memory<int, 1> arr3(sycl::range<1>(2), {1, 2});
static __device__ __constant__ int arr3[2] = {1, 2};

__global__ void g();
