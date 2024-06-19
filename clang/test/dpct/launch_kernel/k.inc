#include <stdexcept>

#define CHECK_2(E)                                               \
  do {                                                           \
    cudaError_t __err = E;                                       \
    if (__err != cudaSuccess) {                                  \
      throw std::runtime_error("");                              \
    }                                                            \
  } while (0)

#define CHECK_1(E) CHECK_2(E)

template <class T>
static __global__ void kernel38() {}

template <class T>
void foo38() {
  void* args[] = {};
  int block;
  int x;
  int y;
  int z;
  int shared;

  //CHECK:dpct::queue_ptr stream;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1123:{{[0-9]+}}: The kernel function pointer cannot be used in the device code. You need to call the kernel function with the correct argument(s) directly. According to the kernel function definition, adjusting the dimension of the sycl::nd_item may also be required.
  //CHECK-NEXT:*/
  //CHECK-NEXT:CHECK_1([&](){
  //CHECK-NEXT:stream->parallel_for(
  //CHECK-NEXT:sycl::nd_range<3>(sycl::range<3>(z, y, x) * sycl::range<3>(1, 1, block), sycl::range<3>(1, 1, block)), 
  //CHECK-NEXT:[=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT: ((void*)&kernel38<T>)();
  //CHECK-NEXT:});
  //CHECK-NEXT:  return 0;
  //CHECK-NEXT:}());
  cudaStream_t stream;
  CHECK_1(cudaLaunchKernel((void*)&kernel38<T>, dim3(x, y, z), block, args, shared, stream));
}
#undef CHECK_1
#undef CHECK_2
