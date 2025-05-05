// RUN: echo "empty command"

// CHECK: void foofunc() {
__global__ void foofunc() {

  // CHECK: size_t tix = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
  size_t tix = threadIdx.x;
}
