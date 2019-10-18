// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sycl_style_double2.dp.cpp --match-full-lines %s

// CHECK: void func3(cl::sycl::double2 a, cl::sycl::double2 b, cl::sycl::double2 c) {
void func3(double2 a, double2 b, double2 c) {
}
// CHECK: void fun(cl::sycl::double2 a) {}
void fun(double2 a) {}

// CHECK: void kernel(cl::sycl::double2* data) {}
__global__ void kernel(double2* data) {}

// CHECK: // Removed.
static __shared__ double2 ctemp2[2]; // Removed.

// CHECK: static void gpuMain(dpct::dpct_accessor<cl::sycl::double2, dpct::local, 1> ctemp2){
// CHECK:   int* ctempi = (int*) (&ctemp2[0]);
// CHECK:   cl::sycl::double2* ctempd =  ctemp2;
// CHECK: }
static __global__ void gpuMain(){
  int* ctempi = (int*) ctemp2;
  double2* ctempd =  ctemp2;
}

int main() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::double2 deflt;
  double2 deflt;

  // CHECK: cl::sycl::double2 copyctor1 = cl::sycl::double2(1, 2);
  double2 copyctor1 = make_double2(1, 2);

  // CHECK: cl::sycl::double2 copyctor2 = cl::sycl::double2(copyctor1);
  double2 copyctor2 = double2(copyctor1);

  // CHECK: cl::sycl::double2 copyctor3(copyctor1);
  double2 copyctor3(copyctor1);

  // CHECK: func3(deflt, cl::sycl::double2(deflt), (cl::sycl::double2)deflt);
  func3(deflt, double2(deflt), (double2)deflt);

  // CHECK: cl::sycl::double2 *i4;
  double2 *i4;
  // CHECK: cl::sycl::double2 *i5;
  double2 *i5;
  // CHECK: cl::sycl::double2 i6;
  double2 i6;
  // CHECK: cl::sycl::double2 i7;
  double2 i7;
  // CHECK: double i = static_cast<double>(i6.x());
  double i = i6.x;
  // CHECK: i6.x() = static_cast<double>(i7.x());
  i6.x = i7.x;
  // CHECK: if (static_cast<double>(i6.x()) == static_cast<double>(i7.x())) {
  if (i6.x == i7.x) {
  }
  // CHECK: cl::sycl::double2 i2_array[10];
  double2 i2_array[10];
  // CHECK: cl::sycl::double2 i2_array2[10];
  double2 i2_array2[10];
  // CHECK: if (static_cast<double>(i2_array[1].x()) == static_cast<double>(i2_array2[1].x())) {
  if (i2_array[1].x == i2_array2[1].x) {
  }
  // CHECK: cl::sycl::double2 x = cl::sycl::double2(1, 2);
  double2 x = make_double2(1, 2);
  // CHECK: i4 = (cl::sycl::double2 *)i2_array;
  i4 = (double2 *)i2_array;
  // CHECK: i7 = (cl::sycl::double2)i6;
  i7 = (double2)i6;
  // CHECK: i7 = cl::sycl::double2(i6);
  i7 = double2(i6);

  // CHECK: cl::sycl::double2* data;
  // CHECK-NEXT: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> data_buf_ct0 = dpct::get_buffer_and_offset(data);
  // CHECK-NEXT:   size_t data_offset_ct0 = data_buf_ct0.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto data_acc_ct0 = data_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::double2 *data_ct0 = (cl::sycl::double2 *)(&data_acc_ct0[0] + data_offset_ct0);
  // CHECK-NEXT:           kernel(data_ct0);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  double2* data;
  kernel<<<1, 1>>>(data);

  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       dpct::dpct_range<1> ctemp2_range_ct1(2);
  // CHECK-NEXT:       cl::sycl::accessor<cl::sycl::double2, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> ctemp2_acc_ct1(ctemp2_range_ct1, cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(64, 1, 1) * cl::sycl::range<3>(64, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(64, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class gpuMain_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           gpuMain(dpct::dpct_accessor<cl::sycl::double2, dpct::local, 1>(ctemp2_acc_ct1, ctemp2_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  gpuMain<<<64, 64>>>();
}
