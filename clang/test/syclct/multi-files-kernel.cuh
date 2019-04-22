
// CHECK: int global_id(cl::sycl::nd_item<3> item_{{[a-f0-9]+}});
__device__ int global_id();


// CHECK: void simple_kernel(int *i_array, cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
__global__ void simple_kernel(int *i_array) {
  int index;
  index = global_id();
  if (index < 360) {
    i_array[index] = index;
  }
  return;
}
