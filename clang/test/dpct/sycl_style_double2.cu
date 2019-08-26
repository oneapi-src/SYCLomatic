// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/sycl_style_double2.dp.cpp --match-full-lines %s

// CHECK: void func3(cl::sycl::double2 a, cl::sycl::double2 b, cl::sycl::double2 c) {
void func3(double2 a, double2 b, double2 c) {
}
// CHECK: void fun(cl::sycl::double2 a) {}
void fun(double2 a) {}

// CHECK: void kernel(cl::sycl::double2* data) {}
__global__ void kernel(double2* data) {}

// CHECK: dpct::local_memory<cl::sycl::double2, 1> ctemp2(2);
static __shared__ double2 ctemp2[2];

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
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(data);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::double2 *arg_ct0 = (cl::sycl::double2 *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           kernel(arg_ct0);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  double2* data;
  kernel<<<1, 1>>>(data);

  // CHECK: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ctemp2_range_ct1 = ctemp2.get_range();
  // CHECK-NEXT:       auto ctemp2_acc_ct1 = ctemp2.get_access(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class gpuMain_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(64, 1, 1) * cl::sycl::range<3>(64, 1, 1)), cl::sycl::range<3>(64, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           gpuMain(dpct::dpct_accessor<cl::sycl::double2, dpct::local, 1>(ctemp2_acc_ct1, ctemp2_range_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  gpuMain<<<64, 64>>>();
}
