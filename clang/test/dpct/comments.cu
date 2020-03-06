// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" --comments -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/comments.dp.cpp

static texture<uint2, 1> tex21;

__constant__ int a = 1;
__device__ int b[36][36];

__device__ void test() {
  __shared__ int c[36];
  c[0] = b[0][0] + a;
}

__global__ void kernel() {
  test();
  __device__ uint2 a[16];
  __shared__ int b[12][12];
  a[0] = tex1D(tex21, 1);
  b[0][0] = 0;
  printf("test\n");
}

int main() {
    // CHECK: // These variables are defined for 3d matrix memory copy.
    // CHECK-NEXT: dpct::pitched_data p_from_data_ct1, p_to_data_ct1;
    // CHECK-NEXT: sycl::id<3> p_from_pos_ct1(0, 0, 0), p_to_pos_ct1(0, 0, 0);
    // CHECK-NEXT: sycl::range<3> p_size_ct1(0, 0, 0);
    // CHECK-NEXT: dpct::memcpy_direction p_direction_ct1;
    cudaMemcpy3DParms p;
    dim3 griddim(1, 2, 3);
    dim3 threaddim(1, 2, 3);

// CHECK:    dpct::get_default_queue().submit(
// CHECK-NEXT:        [&](sycl::handler &cgh) {
// CHECK-NEXT:          sycl::stream stream_ct1(64 * 1024, 80, cgh);
// CHECK-EMPTY:  
// CHECK-NEXT:          dpct::device_memory<sycl::uint2, 1> a(16);
// CHECK-EMPTY:
// CHECK-NEXT:          // ranges used for accessors to device memory
// CHECK-NEXT:          sycl::range<2> b_range_ct1(12, 12);
// CHECK-EMPTY:  
// CHECK-NEXT:          // pointers to device memory
// CHECK-NEXT:          auto a_ptr_ct1 = a.get_ptr();
// CHECK-NEXT:          auto a_ptr_ct1 = a.get_ptr();
// CHECK-EMPTY:  
// CHECK-NEXT:          // accessors to device memory
// CHECK-NEXT:          sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> c_acc_ct1(sycl::range<1>(36), cgh);
// CHECK-NEXT:          sycl::accessor<int, 2, sycl::access::mode::read_write, sycl::access::target::local> b_acc_ct1(b_range_ct1, cgh);
// CHECK-NEXT:          auto b_acc_ct1 = b.get_access(cgh);
// CHECK-EMPTY:  
// CHECK-NEXT:          // accessors to image wrappers
// CHECK-NEXT:          auto tex21_acc = tex21.get_access(cgh);
// CHECK-EMPTY:  
// CHECK-NEXT:          // ranges to define ND iteration space for the kernel
// CHECK-NEXT:          auto dpct_global_range = griddim * threaddim;
// CHECK-EMPTY:  
// CHECK-NEXT:          // run the kernel within defined ND range
// CHECK-NEXT:          cgh.parallel_for(
// CHECK-NEXT:            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
// CHECK-NEXT:            [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:              kernel(stream_ct1, *a_ptr_ct1, b_acc_ct1, c_acc_ct1.get_pointer(), a_ptr_ct1, dpct::accessor<int, dpct::local, 2>(b_acc_ct1, b_range_ct1), tex21_acc);
// CHECK-NEXT:            });
// CHECK-NEXT:        });
    kernel<<<griddim, threaddim>>>();
}