// RUN: dpct --format-range=none --usm-level=none -out-root %T/sycl_style_double2 %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sycl_style_double2/sycl_style_double2.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/sycl_style_double2/sycl_style_double2.dp.cpp -o %T/sycl_style_double2/sycl_style_double2.dp.o %}

// CHECK: void func3(sycl::double2 a, sycl::double2 b, sycl::double2 c) {
void func3(double2 a, double2 b, double2 c) {
}
// CHECK: void fun(sycl::double2 a) {}
void fun(double2 a) {}

// CHECK: void kernel(sycl::double2* data) {
__global__ void kernel(double2* data) {
}

// CHECK: // Removed.
static __shared__ double2 ctemp2[2]; // Removed.

// CHECK: static void gpuMain(sycl::double2 *ctemp2){
// CHECK:   int* ctempi = (int*) ctemp2;
// CHECK:   sycl::double2* ctempd =  ctemp2;
// CHECK: }
static __global__ void gpuMain(){
  int* ctempi = (int*) ctemp2;
  double2* ctempd =  ctemp2;
}

int main() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
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
  // CHECK-NEXT: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper data_acc_ct0(data, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel(data_acc_ct0.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  double2* data;
  kernel<<<1, 1>>>(data);

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       sycl::local_accessor<sycl::double2, 1> ctemp2_acc_ct1(sycl::range<1>(2), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class gpuMain_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 64) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           gpuMain(ctemp2_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  gpuMain<<<64, 64>>>();
}

