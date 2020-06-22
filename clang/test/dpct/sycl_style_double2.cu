// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sycl_style_double2.dp.cpp --match-full-lines %s

// CHECK: void func3(sycl::double2 a, sycl::double2 b, sycl::double2 c) {
void func3(double2 a, double2 b, double2 c) {
}
// CHECK: void fun(sycl::double2 a) {}
void fun(double2 a) {}

// CHECK: void kernel(sycl::double2* data) {}
__global__ void kernel(double2* data) {}

// CHECK: // Removed.
static __shared__ double2 ctemp2[2]; // Removed.

// CHECK: static void gpuMain(sycl::double2 *ctemp2){
// CHECK:   int* ctempi = (int*) (&ctemp2[0]);
// CHECK:   sycl::double2* ctempd =  ctemp2;
// CHECK: }
static __global__ void gpuMain(){
  int* ctempi = (int*) ctemp2;
  double2* ctempd =  ctemp2;
}

int main() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // range default constructor does the right thing.
  // CHECK: sycl::double2 deflt;
  double2 deflt;

  // CHECK: sycl::double2 copyctor1 = sycl::double2(1, 2);
  double2 copyctor1 = make_double2(1, 2);

  // CHECK: sycl::double2 copyctor2 = sycl::double2(copyctor1);
  double2 copyctor2 = double2(copyctor1);

  // CHECK: sycl::double2 copyctor3(copyctor1);
  double2 copyctor3(copyctor1);

  // CHECK: func3(deflt, sycl::double2(deflt), (sycl::double2)deflt);
  func3(deflt, double2(deflt), (double2)deflt);

  // CHECK: sycl::double2 *i4;
  double2 *i4;
  // CHECK: sycl::double2 *i5;
  double2 *i5;
  // CHECK: sycl::double2 i6;
  double2 i6;
  // CHECK: sycl::double2 i7;
  double2 i7;
  // CHECK: double i = i6.x();
  double i = i6.x;
  // CHECK: i6.x() = i7.x();
  i6.x = i7.x;
  // CHECK: if (i6.x() == i7.x()) {
  if (i6.x == i7.x) {
  }
  // CHECK: sycl::double2 i2_array[10];
  double2 i2_array[10];
  // CHECK: sycl::double2 i2_array2[10];
  double2 i2_array2[10];
  // CHECK: if (i2_array[1].x() == i2_array2[1].x()) {
  if (i2_array[1].x == i2_array2[1].x) {
  }
  // CHECK: sycl::double2 x = sycl::double2(1, 2);
  double2 x = make_double2(1, 2);
  // CHECK: i4 = (sycl::double2 *)i2_array;
  i4 = (double2 *)i2_array;
  // CHECK: i7 = (sycl::double2)i6;
  i7 = (double2)i6;
  // CHECK: i7 = sycl::double2(i6);
  i7 = double2(i6);

  // CHECK: sycl::double2* data;
  // CHECK-NEXT: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> data_buf_ct0 = dpct::get_buffer_and_offset(data);
  // CHECK-NEXT:   size_t data_offset_ct0 = data_buf_ct0.second;
  // CHECK-NEXT:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto data_acc_ct0 = data_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           sycl::double2 *data_ct0 = (sycl::double2 *)(&data_acc_ct0[0] + data_offset_ct0);
  // CHECK-NEXT:           kernel(data_ct0);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  double2* data;
  kernel<<<1, 1>>>(data);

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       sycl::accessor<sycl::double2, 1, sycl::access::mode::read_write, sycl::access::target::local> ctemp2_acc_ct1(sycl::range<1>(2), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class gpuMain_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 64) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           gpuMain(ctemp2_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  gpuMain<<<64, 64>>>();
}
