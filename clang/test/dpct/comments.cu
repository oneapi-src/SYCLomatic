// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6
// RUN: dpct --format-range=none -out-root %T/comments %s --cuda-include-path="%cuda-path/include" --comments -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/comments/comments.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/comments/comments.dp.cpp -o %T/comments/comments.dp.o %}

static texture<uint2, 1> tex21;

__constant__ int a = 1;
__device__ int b[36][36];

__device__ void test() {
  __shared__ int cl[36];
  cl[0] = b[0][0] + a;
}

__device__ uint2 al[16];
__global__ void kernel() {
  test();
  __shared__ int bl[12][12];
  al[0] = tex1D(tex21, 1);
  bl[0][0] = 0;
  printf("test\n");
}

int main() {
    dim3 griddim(1, 2, 3);
    dim3 threaddim(1, 2, 3);

//      CHECK:    {
// CHECK-NEXT:      // init global memory
// CHECK-NEXT:      a.init();
// CHECK-NEXT:      b.init();
// CHECK-NEXT:      al.init();
// CHECK-NEXT:      tex21.create_image();
// CHECK-EMPTY:
// CHECK-NEXT:      dpct::get_in_order_queue().submit(
// CHECK-NEXT:          [&](sycl::handler &cgh) {
// CHECK-NEXT:            sycl::stream stream_ct1(64 * 1024, 80, cgh);
// CHECK-EMPTY:  
// CHECK-NEXT:            // pointers to device memory
// CHECK-NEXT:            auto a_ptr_ct1 = a.get_ptr();
// CHECK-NEXT:            auto al_ptr_ct1 = al.get_ptr();
// CHECK-EMPTY:  
// CHECK-NEXT:            // accessors to device memory
// CHECK-NEXT:            sycl::local_accessor<int, 1> cl_acc_ct1(sycl::range<1>(36), cgh);
// CHECK-NEXT:            sycl::local_accessor<int[12][12], 0> bl_acc_ct1(cgh);
// CHECK-NEXT:            auto b_acc_ct1 = b.get_access(cgh);
// CHECK-EMPTY:  
// CHECK-NEXT:            // accessors to image objects
// CHECK-NEXT:            auto tex21_acc = tex21.get_access(cgh);
// CHECK-EMPTY:  
// CHECK-NEXT:            // sampler of image objects
// CHECK-NEXT:            auto tex21_smpl = tex21.get_sampler();
// CHECK-EMPTY:  
// CHECK-NEXT:            cgh.parallel_for(
// CHECK-NEXT:              sycl::nd_range<3>(griddim * threaddim, threaddim),
// CHECK-NEXT:              [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:                kernel(stream_ct1, *a_ptr_ct1, b_acc_ct1, al_ptr_ct1, cl_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(), bl_acc_ct1, dpct::image_accessor_ext<sycl::uint2, 1>(tex21_smpl, tex21_acc));
// CHECK-NEXT:              });
// CHECK-NEXT:          });
// CHECK-NEXT:    }
    kernel<<<griddim, threaddim>>>();
}

