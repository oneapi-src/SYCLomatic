// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/vector_type.dp.cpp --match-full-lines %s

// CHECK: void func3_char1(char a, char b, char c) {
void func3_char1(char1 a, char1 b, char1 c) {
}
// CHECK: void func_char1(char a) {
void func_char1(char1 a) {
}
// CHECK: void kernel_char1(char *a, char *b) {
__global__ void kernel_char1(char1 *a, char1 *b) {
}

int main_char1() {
  // range default constructor does the right thing.
  // CHECK: char char1_a;
  char1 char1_a;
  // CHECK: char char1_b = char(1);
  char1 char1_b = make_char1(1);
  // CHECK: char char1_c = char(char1_b);
  char1 char1_c = char1(char1_b);
  // CHECK: char char1_d(char1_c);
  char1 char1_d(char1_c);
  // CHECK: func3_char1(char1_b, char(char1_b), (char)char1_b);
  func3_char1(char1_b, char1(char1_b), (char1)char1_b);
  // CHECK: char *char1_e;
  char1 *char1_e;
  // CHECK: char *char1_f;
  char1 *char1_f;
  // CHECK: signed char char1_g = static_cast<signed char>(char1_c);
  signed char char1_g = char1_c.x;
  // CHECK: char1_a = static_cast<signed char>(char1_d);
  char1_a.x = char1_d.x;
  // CHECK: if (static_cast<signed char>(char1_b) == static_cast<signed char>(char1_d)) {}
  if (char1_b.x == char1_d.x) {}
  // CHECK: char char1_h[16];
  char1 char1_h[16];
  // CHECK: char char1_i[32];
  char1 char1_i[32];
  // CHECK: if (static_cast<signed char>(char1_h[12]) == static_cast<signed char>(char1_i[12])) {}
  if (char1_h[12].x == char1_i[12].x) {}
  // CHECK: char1_f = (char *)char1_i;
  char1_f = (char1 *)char1_i;
  // CHECK: char1_a = (char)char1_c;
  char1_a = (char1)char1_c;
  // CHECK: char1_b = char(char1_b);
  char1_b = char1(char1_b);
  // CHECK: char char1_j, char1_k, char1_l, char1_m[16], *char1_n[32];
  char1 char1_j, char1_k, char1_l, char1_m[16], *char1_n[32];
  // CHECK: int char1_o = sizeof(char);
  int char1_o = sizeof(char1);
  // CHECK: int signed char_p = sizeof(signed char);
  int signed char_p = sizeof(signed char);
  // CHECK: int char1_q = sizeof(char1_d);
  int char1_q = sizeof(char1_d);
  int *char1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> char1_e_buf_ct0 = dpct::get_buffer_and_offset(char1_e);
  // CHECK-NEXT:   size_t char1_e_offset_ct0 = char1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> char1_cast_buf_ct1 = dpct::get_buffer_and_offset((char *)char1_cast);
  // CHECK-NEXT:   size_t char1_cast_offset_ct1 = char1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto char1_e_acc_ct0 = char1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto char1_cast_acc_ct1 = char1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_char1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           char *char1_e_ct0 = (char *)(&char1_e_acc_ct0[0] + char1_e_offset_ct0);
  // CHECK-NEXT:           char *char1_cast_ct1 = (char *)(&char1_cast_acc_ct1[0] + char1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_char1(char1_e_ct0, char1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_char1<<<1,1>>>(char1_e, (char1 *)char1_cast);
  return 0;
}

// CHECK: void func3_char2(cl::sycl::char2 a, cl::sycl::char2 b, cl::sycl::char2 c) {
void func3_char2(char2 a, char2 b, char2 c) {
}
// CHECK: void func_char2(cl::sycl::char2 a) {
void func_char2(char2 a) {
}
// CHECK: void kernel_char2(cl::sycl::char2 *a, cl::sycl::char2 *b) {
__global__ void kernel_char2(char2 *a, char2 *b) {
}

int main_char2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::char2 char2_a;
  char2 char2_a;
  // CHECK: cl::sycl::char2 char2_b = cl::sycl::char2(1, 2);
  char2 char2_b = make_char2(1, 2);
  // CHECK: cl::sycl::char2 char2_c = cl::sycl::char2(char2_b);
  char2 char2_c = char2(char2_b);
  // CHECK: cl::sycl::char2 char2_d(char2_c);
  char2 char2_d(char2_c);
  // CHECK: func3_char2(char2_b, cl::sycl::char2(char2_b), (cl::sycl::char2)char2_b);
  func3_char2(char2_b, char2(char2_b), (char2)char2_b);
  // CHECK: cl::sycl::char2 *char2_e;
  char2 *char2_e;
  // CHECK: cl::sycl::char2 *char2_f;
  char2 *char2_f;
  // CHECK: signed char char2_g = static_cast<signed char>(char2_c.x());
  signed char char2_g = char2_c.x;
  // CHECK: char2_a.x() = static_cast<signed char>(char2_d.x());
  char2_a.x = char2_d.x;
  // CHECK: if (static_cast<signed char>(char2_b.x()) == static_cast<signed char>(char2_d.x())) {}
  if (char2_b.x == char2_d.x) {}
  // CHECK: cl::sycl::char2 char2_h[16];
  char2 char2_h[16];
  // CHECK: cl::sycl::char2 char2_i[32];
  char2 char2_i[32];
  // CHECK: if (static_cast<signed char>(char2_h[12].x()) == static_cast<signed char>(char2_i[12].x())) {}
  if (char2_h[12].x == char2_i[12].x) {}
  // CHECK: char2_f = (cl::sycl::char2 *)char2_i;
  char2_f = (char2 *)char2_i;
  // CHECK: char2_a = (cl::sycl::char2)char2_c;
  char2_a = (char2)char2_c;
  // CHECK: char2_b = cl::sycl::char2(char2_b);
  char2_b = char2(char2_b);
  // CHECK: cl::sycl::char2 char2_j, char2_k, char2_l, char2_m[16], *char2_n[32];
  char2 char2_j, char2_k, char2_l, char2_m[16], *char2_n[32];
  // CHECK: int char2_o = sizeof(cl::sycl::char2);
  int char2_o = sizeof(char2);
  // CHECK: int signed char_p = sizeof(signed char);
  int signed char_p = sizeof(signed char);
  // CHECK: int char2_q = sizeof(char2_d);
  int char2_q = sizeof(char2_d);
  int *char2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> char2_e_buf_ct0 = dpct::get_buffer_and_offset(char2_e);
  // CHECK-NEXT:   size_t char2_e_offset_ct0 = char2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> char2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::char2 *)char2_cast);
  // CHECK-NEXT:   size_t char2_cast_offset_ct1 = char2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto char2_e_acc_ct0 = char2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto char2_cast_acc_ct1 = char2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_char2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::char2 *char2_e_ct0 = (cl::sycl::char2 *)(&char2_e_acc_ct0[0] + char2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::char2 *char2_cast_ct1 = (cl::sycl::char2 *)(&char2_cast_acc_ct1[0] + char2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_char2(char2_e_ct0, char2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_char2<<<1,1>>>(char2_e, (char2 *)char2_cast);
  return 0;
}

// CHECK: void func3_char3(cl::sycl::char3 a, cl::sycl::char3 b, cl::sycl::char3 c) {
void func3_char3(char3 a, char3 b, char3 c) {
}
// CHECK: void func_char3(cl::sycl::char3 a) {
void func_char3(char3 a) {
}
// CHECK: void kernel_char3(cl::sycl::char3 *a, cl::sycl::char3 *b) {
__global__ void kernel_char3(char3 *a, char3 *b) {
}

int main_char3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::char3 char3_a;
  char3 char3_a;
  // CHECK: cl::sycl::char3 char3_b = cl::sycl::char3(1, 2, 3);
  char3 char3_b = make_char3(1, 2, 3);
  // CHECK: cl::sycl::char3 char3_c = cl::sycl::char3(char3_b);
  char3 char3_c = char3(char3_b);
  // CHECK: cl::sycl::char3 char3_d(char3_c);
  char3 char3_d(char3_c);
  // CHECK: func3_char3(char3_b, cl::sycl::char3(char3_b), (cl::sycl::char3)char3_b);
  func3_char3(char3_b, char3(char3_b), (char3)char3_b);
  // CHECK: cl::sycl::char3 *char3_e;
  char3 *char3_e;
  // CHECK: cl::sycl::char3 *char3_f;
  char3 *char3_f;
  // CHECK: signed char char3_g = static_cast<signed char>(char3_c.x());
  signed char char3_g = char3_c.x;
  // CHECK: char3_a.x() = static_cast<signed char>(char3_d.x());
  char3_a.x = char3_d.x;
  // CHECK: if (static_cast<signed char>(char3_b.x()) == static_cast<signed char>(char3_d.x())) {}
  if (char3_b.x == char3_d.x) {}
  // CHECK: cl::sycl::char3 char3_h[16];
  char3 char3_h[16];
  // CHECK: cl::sycl::char3 char3_i[32];
  char3 char3_i[32];
  // CHECK: if (static_cast<signed char>(char3_h[12].x()) == static_cast<signed char>(char3_i[12].x())) {}
  if (char3_h[12].x == char3_i[12].x) {}
  // CHECK: char3_f = (cl::sycl::char3 *)char3_i;
  char3_f = (char3 *)char3_i;
  // CHECK: char3_a = (cl::sycl::char3)char3_c;
  char3_a = (char3)char3_c;
  // CHECK: char3_b = cl::sycl::char3(char3_b);
  char3_b = char3(char3_b);
  // CHECK: cl::sycl::char3 char3_j, char3_k, char3_l, char3_m[16], *char3_n[32];
  char3 char3_j, char3_k, char3_l, char3_m[16], *char3_n[32];
  // CHECK: int char3_o = sizeof(cl::sycl::char3);
  int char3_o = sizeof(char3);
  // CHECK: int signed char_p = sizeof(signed char);
  int signed char_p = sizeof(signed char);
  // CHECK: int char3_q = sizeof(char3_d);
  int char3_q = sizeof(char3_d);
  int *char3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> char3_e_buf_ct0 = dpct::get_buffer_and_offset(char3_e);
  // CHECK-NEXT:   size_t char3_e_offset_ct0 = char3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> char3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::char3 *)char3_cast);
  // CHECK-NEXT:   size_t char3_cast_offset_ct1 = char3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto char3_e_acc_ct0 = char3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto char3_cast_acc_ct1 = char3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_char3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::char3 *char3_e_ct0 = (cl::sycl::char3 *)(&char3_e_acc_ct0[0] + char3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::char3 *char3_cast_ct1 = (cl::sycl::char3 *)(&char3_cast_acc_ct1[0] + char3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_char3(char3_e_ct0, char3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_char3<<<1,1>>>(char3_e, (char3 *)char3_cast);
  return 0;
}

// CHECK: void func3_char4(cl::sycl::char4 a, cl::sycl::char4 b, cl::sycl::char4 c) {
void func3_char4(char4 a, char4 b, char4 c) {
}
// CHECK: void func_char4(cl::sycl::char4 a) {
void func_char4(char4 a) {
}
// CHECK: void kernel_char4(cl::sycl::char4 *a, cl::sycl::char4 *b) {
__global__ void kernel_char4(char4 *a, char4 *b) {
}

int main_char4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::char4 char4_a;
  char4 char4_a;
  // CHECK: cl::sycl::char4 char4_b = cl::sycl::char4(1, 2, 3, 4);
  char4 char4_b = make_char4(1, 2, 3, 4);
  // CHECK: cl::sycl::char4 char4_c = cl::sycl::char4(char4_b);
  char4 char4_c = char4(char4_b);
  // CHECK: cl::sycl::char4 char4_d(char4_c);
  char4 char4_d(char4_c);
  // CHECK: func3_char4(char4_b, cl::sycl::char4(char4_b), (cl::sycl::char4)char4_b);
  func3_char4(char4_b, char4(char4_b), (char4)char4_b);
  // CHECK: cl::sycl::char4 *char4_e;
  char4 *char4_e;
  // CHECK: cl::sycl::char4 *char4_f;
  char4 *char4_f;
  // CHECK: signed char char4_g = static_cast<signed char>(char4_c.x());
  signed char char4_g = char4_c.x;
  // CHECK: char4_a.x() = static_cast<signed char>(char4_d.x());
  char4_a.x = char4_d.x;
  // CHECK: if (static_cast<signed char>(char4_b.x()) == static_cast<signed char>(char4_d.x())) {}
  if (char4_b.x == char4_d.x) {}
  // CHECK: cl::sycl::char4 char4_h[16];
  char4 char4_h[16];
  // CHECK: cl::sycl::char4 char4_i[32];
  char4 char4_i[32];
  // CHECK: if (static_cast<signed char>(char4_h[12].x()) == static_cast<signed char>(char4_i[12].x())) {}
  if (char4_h[12].x == char4_i[12].x) {}
  // CHECK: char4_f = (cl::sycl::char4 *)char4_i;
  char4_f = (char4 *)char4_i;
  // CHECK: char4_a = (cl::sycl::char4)char4_c;
  char4_a = (char4)char4_c;
  // CHECK: char4_b = cl::sycl::char4(char4_b);
  char4_b = char4(char4_b);
  // CHECK: cl::sycl::char4 char4_j, char4_k, char4_l, char4_m[16], *char4_n[32];
  char4 char4_j, char4_k, char4_l, char4_m[16], *char4_n[32];
  // CHECK: int char4_o = sizeof(cl::sycl::char4);
  int char4_o = sizeof(char4);
  // CHECK: int signed char_p = sizeof(signed char);
  int signed char_p = sizeof(signed char);
  // CHECK: int char4_q = sizeof(char4_d);
  int char4_q = sizeof(char4_d);
  int *char4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> char4_e_buf_ct0 = dpct::get_buffer_and_offset(char4_e);
  // CHECK-NEXT:   size_t char4_e_offset_ct0 = char4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> char4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::char4 *)char4_cast);
  // CHECK-NEXT:   size_t char4_cast_offset_ct1 = char4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto char4_e_acc_ct0 = char4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto char4_cast_acc_ct1 = char4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_char4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::char4 *char4_e_ct0 = (cl::sycl::char4 *)(&char4_e_acc_ct0[0] + char4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::char4 *char4_cast_ct1 = (cl::sycl::char4 *)(&char4_cast_acc_ct1[0] + char4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_char4(char4_e_ct0, char4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_char4<<<1,1>>>(char4_e, (char4 *)char4_cast);
  return 0;
}

// CHECK: void func3_double1(double a, double b, double c) {
void func3_double1(double1 a, double1 b, double1 c) {
}
// CHECK: void func_double1(double a) {
void func_double1(double1 a) {
}
// CHECK: void kernel_double1(double *a, double *b) {
__global__ void kernel_double1(double1 *a, double1 *b) {
}

int main_double1() {
  // range default constructor does the right thing.
  // CHECK: double double1_a;
  double1 double1_a;
  // CHECK: double double1_b = double(1);
  double1 double1_b = make_double1(1);
  // CHECK: double double1_c = double(double1_b);
  double1 double1_c = double1(double1_b);
  // CHECK: double double1_d(double1_c);
  double1 double1_d(double1_c);
  // CHECK: func3_double1(double1_b, double(double1_b), (double)double1_b);
  func3_double1(double1_b, double1(double1_b), (double1)double1_b);
  // CHECK: double *double1_e;
  double1 *double1_e;
  // CHECK: double *double1_f;
  double1 *double1_f;
  // CHECK: double double1_g = static_cast<double>(double1_c);
  double double1_g = double1_c.x;
  // CHECK: double1_a = static_cast<double>(double1_d);
  double1_a.x = double1_d.x;
  // CHECK: if (static_cast<double>(double1_b) == static_cast<double>(double1_d)) {}
  if (double1_b.x == double1_d.x) {}
  // CHECK: double double1_h[16];
  double1 double1_h[16];
  // CHECK: double double1_i[32];
  double1 double1_i[32];
  // CHECK: if (static_cast<double>(double1_h[12]) == static_cast<double>(double1_i[12])) {}
  if (double1_h[12].x == double1_i[12].x) {}
  // CHECK: double1_f = (double *)double1_i;
  double1_f = (double1 *)double1_i;
  // CHECK: double1_a = (double)double1_c;
  double1_a = (double1)double1_c;
  // CHECK: double1_b = double(double1_b);
  double1_b = double1(double1_b);
  // CHECK: double double1_j, double1_k, double1_l, double1_m[16], *double1_n[32];
  double1 double1_j, double1_k, double1_l, double1_m[16], *double1_n[32];
  // CHECK: int double1_o = sizeof(double);
  int double1_o = sizeof(double1);
  // CHECK: int double_p = sizeof(double);
  int double_p = sizeof(double);
  // CHECK: int double1_q = sizeof(double1_d);
  int double1_q = sizeof(double1_d);
  int *double1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> double1_e_buf_ct0 = dpct::get_buffer_and_offset(double1_e);
  // CHECK-NEXT:   size_t double1_e_offset_ct0 = double1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> double1_cast_buf_ct1 = dpct::get_buffer_and_offset((double *)double1_cast);
  // CHECK-NEXT:   size_t double1_cast_offset_ct1 = double1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto double1_e_acc_ct0 = double1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto double1_cast_acc_ct1 = double1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_double1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           double *double1_e_ct0 = (double *)(&double1_e_acc_ct0[0] + double1_e_offset_ct0);
  // CHECK-NEXT:           double *double1_cast_ct1 = (double *)(&double1_cast_acc_ct1[0] + double1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_double1(double1_e_ct0, double1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_double1<<<1,1>>>(double1_e, (double1 *)double1_cast);
  return 0;
}

// CHECK: void func3_double2(cl::sycl::double2 a, cl::sycl::double2 b, cl::sycl::double2 c) {
void func3_double2(double2 a, double2 b, double2 c) {
}
// CHECK: void func_double2(cl::sycl::double2 a) {
void func_double2(double2 a) {
}
// CHECK: void kernel_double2(cl::sycl::double2 *a, cl::sycl::double2 *b) {
__global__ void kernel_double2(double2 *a, double2 *b) {
}

int main_double2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::double2 double2_a;
  double2 double2_a;
  // CHECK: cl::sycl::double2 double2_b = cl::sycl::double2(1, 2);
  double2 double2_b = make_double2(1, 2);
  // CHECK: cl::sycl::double2 double2_c = cl::sycl::double2(double2_b);
  double2 double2_c = double2(double2_b);
  // CHECK: cl::sycl::double2 double2_d(double2_c);
  double2 double2_d(double2_c);
  // CHECK: func3_double2(double2_b, cl::sycl::double2(double2_b), (cl::sycl::double2)double2_b);
  func3_double2(double2_b, double2(double2_b), (double2)double2_b);
  // CHECK: cl::sycl::double2 *double2_e;
  double2 *double2_e;
  // CHECK: cl::sycl::double2 *double2_f;
  double2 *double2_f;
  // CHECK: double double2_g = static_cast<double>(double2_c.x());
  double double2_g = double2_c.x;
  // CHECK: double2_a.x() = static_cast<double>(double2_d.x());
  double2_a.x = double2_d.x;
  // CHECK: if (static_cast<double>(double2_b.x()) == static_cast<double>(double2_d.x())) {}
  if (double2_b.x == double2_d.x) {}
  // CHECK: cl::sycl::double2 double2_h[16];
  double2 double2_h[16];
  // CHECK: cl::sycl::double2 double2_i[32];
  double2 double2_i[32];
  // CHECK: if (static_cast<double>(double2_h[12].x()) == static_cast<double>(double2_i[12].x())) {}
  if (double2_h[12].x == double2_i[12].x) {}
  // CHECK: double2_f = (cl::sycl::double2 *)double2_i;
  double2_f = (double2 *)double2_i;
  // CHECK: double2_a = (cl::sycl::double2)double2_c;
  double2_a = (double2)double2_c;
  // CHECK: double2_b = cl::sycl::double2(double2_b);
  double2_b = double2(double2_b);
  // CHECK: cl::sycl::double2 double2_j, double2_k, double2_l, double2_m[16], *double2_n[32];
  double2 double2_j, double2_k, double2_l, double2_m[16], *double2_n[32];
  // CHECK: int double2_o = sizeof(cl::sycl::double2);
  int double2_o = sizeof(double2);
  // CHECK: int double_p = sizeof(double);
  int double_p = sizeof(double);
  // CHECK: int double2_q = sizeof(double2_d);
  int double2_q = sizeof(double2_d);
  int *double2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> double2_e_buf_ct0 = dpct::get_buffer_and_offset(double2_e);
  // CHECK-NEXT:   size_t double2_e_offset_ct0 = double2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> double2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::double2 *)double2_cast);
  // CHECK-NEXT:   size_t double2_cast_offset_ct1 = double2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto double2_e_acc_ct0 = double2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto double2_cast_acc_ct1 = double2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_double2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::double2 *double2_e_ct0 = (cl::sycl::double2 *)(&double2_e_acc_ct0[0] + double2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::double2 *double2_cast_ct1 = (cl::sycl::double2 *)(&double2_cast_acc_ct1[0] + double2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_double2(double2_e_ct0, double2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_double2<<<1,1>>>(double2_e, (double2 *)double2_cast);
  return 0;
}

// CHECK: void func3_double3(cl::sycl::double3 a, cl::sycl::double3 b, cl::sycl::double3 c) {
void func3_double3(double3 a, double3 b, double3 c) {
}
// CHECK: void func_double3(cl::sycl::double3 a) {
void func_double3(double3 a) {
}
// CHECK: void kernel_double3(cl::sycl::double3 *a, cl::sycl::double3 *b) {
__global__ void kernel_double3(double3 *a, double3 *b) {
}

int main_double3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::double3 double3_a;
  double3 double3_a;
  // CHECK: cl::sycl::double3 double3_b = cl::sycl::double3(1, 2, 3);
  double3 double3_b = make_double3(1, 2, 3);
  // CHECK: cl::sycl::double3 double3_c = cl::sycl::double3(double3_b);
  double3 double3_c = double3(double3_b);
  // CHECK: cl::sycl::double3 double3_d(double3_c);
  double3 double3_d(double3_c);
  // CHECK: func3_double3(double3_b, cl::sycl::double3(double3_b), (cl::sycl::double3)double3_b);
  func3_double3(double3_b, double3(double3_b), (double3)double3_b);
  // CHECK: cl::sycl::double3 *double3_e;
  double3 *double3_e;
  // CHECK: cl::sycl::double3 *double3_f;
  double3 *double3_f;
  // CHECK: double double3_g = static_cast<double>(double3_c.x());
  double double3_g = double3_c.x;
  // CHECK: double3_a.x() = static_cast<double>(double3_d.x());
  double3_a.x = double3_d.x;
  // CHECK: if (static_cast<double>(double3_b.x()) == static_cast<double>(double3_d.x())) {}
  if (double3_b.x == double3_d.x) {}
  // CHECK: cl::sycl::double3 double3_h[16];
  double3 double3_h[16];
  // CHECK: cl::sycl::double3 double3_i[32];
  double3 double3_i[32];
  // CHECK: if (static_cast<double>(double3_h[12].x()) == static_cast<double>(double3_i[12].x())) {}
  if (double3_h[12].x == double3_i[12].x) {}
  // CHECK: double3_f = (cl::sycl::double3 *)double3_i;
  double3_f = (double3 *)double3_i;
  // CHECK: double3_a = (cl::sycl::double3)double3_c;
  double3_a = (double3)double3_c;
  // CHECK: double3_b = cl::sycl::double3(double3_b);
  double3_b = double3(double3_b);
  // CHECK: cl::sycl::double3 double3_j, double3_k, double3_l, double3_m[16], *double3_n[32];
  double3 double3_j, double3_k, double3_l, double3_m[16], *double3_n[32];
  // CHECK: int double3_o = sizeof(cl::sycl::double3);
  int double3_o = sizeof(double3);
  // CHECK: int double_p = sizeof(double);
  int double_p = sizeof(double);
  // CHECK: int double3_q = sizeof(double3_d);
  int double3_q = sizeof(double3_d);
  int *double3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> double3_e_buf_ct0 = dpct::get_buffer_and_offset(double3_e);
  // CHECK-NEXT:   size_t double3_e_offset_ct0 = double3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> double3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::double3 *)double3_cast);
  // CHECK-NEXT:   size_t double3_cast_offset_ct1 = double3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto double3_e_acc_ct0 = double3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto double3_cast_acc_ct1 = double3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_double3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::double3 *double3_e_ct0 = (cl::sycl::double3 *)(&double3_e_acc_ct0[0] + double3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::double3 *double3_cast_ct1 = (cl::sycl::double3 *)(&double3_cast_acc_ct1[0] + double3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_double3(double3_e_ct0, double3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_double3<<<1,1>>>(double3_e, (double3 *)double3_cast);
  return 0;
}

// CHECK: void func3_double4(cl::sycl::double4 a, cl::sycl::double4 b, cl::sycl::double4 c) {
void func3_double4(double4 a, double4 b, double4 c) {
}
// CHECK: void func_double4(cl::sycl::double4 a) {
void func_double4(double4 a) {
}
// CHECK: void kernel_double4(cl::sycl::double4 *a, cl::sycl::double4 *b) {
__global__ void kernel_double4(double4 *a, double4 *b) {
}

int main_double4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::double4 double4_a;
  double4 double4_a;
  // CHECK: cl::sycl::double4 double4_b = cl::sycl::double4(1, 2, 3, 4);
  double4 double4_b = make_double4(1, 2, 3, 4);
  // CHECK: cl::sycl::double4 double4_c = cl::sycl::double4(double4_b);
  double4 double4_c = double4(double4_b);
  // CHECK: cl::sycl::double4 double4_d(double4_c);
  double4 double4_d(double4_c);
  // CHECK: func3_double4(double4_b, cl::sycl::double4(double4_b), (cl::sycl::double4)double4_b);
  func3_double4(double4_b, double4(double4_b), (double4)double4_b);
  // CHECK: cl::sycl::double4 *double4_e;
  double4 *double4_e;
  // CHECK: cl::sycl::double4 *double4_f;
  double4 *double4_f;
  // CHECK: double double4_g = static_cast<double>(double4_c.x());
  double double4_g = double4_c.x;
  // CHECK: double4_a.x() = static_cast<double>(double4_d.x());
  double4_a.x = double4_d.x;
  // CHECK: if (static_cast<double>(double4_b.x()) == static_cast<double>(double4_d.x())) {}
  if (double4_b.x == double4_d.x) {}
  // CHECK: cl::sycl::double4 double4_h[16];
  double4 double4_h[16];
  // CHECK: cl::sycl::double4 double4_i[32];
  double4 double4_i[32];
  // CHECK: if (static_cast<double>(double4_h[12].x()) == static_cast<double>(double4_i[12].x())) {}
  if (double4_h[12].x == double4_i[12].x) {}
  // CHECK: double4_f = (cl::sycl::double4 *)double4_i;
  double4_f = (double4 *)double4_i;
  // CHECK: double4_a = (cl::sycl::double4)double4_c;
  double4_a = (double4)double4_c;
  // CHECK: double4_b = cl::sycl::double4(double4_b);
  double4_b = double4(double4_b);
  // CHECK: cl::sycl::double4 double4_j, double4_k, double4_l, double4_m[16], *double4_n[32];
  double4 double4_j, double4_k, double4_l, double4_m[16], *double4_n[32];
  // CHECK: int double4_o = sizeof(cl::sycl::double4);
  int double4_o = sizeof(double4);
  // CHECK: int double_p = sizeof(double);
  int double_p = sizeof(double);
  // CHECK: int double4_q = sizeof(double4_d);
  int double4_q = sizeof(double4_d);
  int *double4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> double4_e_buf_ct0 = dpct::get_buffer_and_offset(double4_e);
  // CHECK-NEXT:   size_t double4_e_offset_ct0 = double4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> double4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::double4 *)double4_cast);
  // CHECK-NEXT:   size_t double4_cast_offset_ct1 = double4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto double4_e_acc_ct0 = double4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto double4_cast_acc_ct1 = double4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_double4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::double4 *double4_e_ct0 = (cl::sycl::double4 *)(&double4_e_acc_ct0[0] + double4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::double4 *double4_cast_ct1 = (cl::sycl::double4 *)(&double4_cast_acc_ct1[0] + double4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_double4(double4_e_ct0, double4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_double4<<<1,1>>>(double4_e, (double4 *)double4_cast);
  return 0;
}

// CHECK: void func3_float1(float a, float b, float c) {
void func3_float1(float1 a, float1 b, float1 c) {
}
// CHECK: void func_float1(float a) {
void func_float1(float1 a) {
}
// CHECK: void kernel_float1(float *a, float *b) {
__global__ void kernel_float1(float1 *a, float1 *b) {
}

int main_float1() {
  // range default constructor does the right thing.
  // CHECK: float float1_a;
  float1 float1_a;
  // CHECK: float float1_b = float(1);
  float1 float1_b = make_float1(1);
  // CHECK: float float1_c = float(float1_b);
  float1 float1_c = float1(float1_b);
  // CHECK: float float1_d(float1_c);
  float1 float1_d(float1_c);
  // CHECK: func3_float1(float1_b, float(float1_b), (float)float1_b);
  func3_float1(float1_b, float1(float1_b), (float1)float1_b);
  // CHECK: float *float1_e;
  float1 *float1_e;
  // CHECK: float *float1_f;
  float1 *float1_f;
  // CHECK: float float1_g = static_cast<float>(float1_c);
  float float1_g = float1_c.x;
  // CHECK: float1_a = static_cast<float>(float1_d);
  float1_a.x = float1_d.x;
  // CHECK: if (static_cast<float>(float1_b) == static_cast<float>(float1_d)) {}
  if (float1_b.x == float1_d.x) {}
  // CHECK: float float1_h[16];
  float1 float1_h[16];
  // CHECK: float float1_i[32];
  float1 float1_i[32];
  // CHECK: if (static_cast<float>(float1_h[12]) == static_cast<float>(float1_i[12])) {}
  if (float1_h[12].x == float1_i[12].x) {}
  // CHECK: float1_f = (float *)float1_i;
  float1_f = (float1 *)float1_i;
  // CHECK: float1_a = (float)float1_c;
  float1_a = (float1)float1_c;
  // CHECK: float1_b = float(float1_b);
  float1_b = float1(float1_b);
  // CHECK: float float1_j, float1_k, float1_l, float1_m[16], *float1_n[32];
  float1 float1_j, float1_k, float1_l, float1_m[16], *float1_n[32];
  // CHECK: int float1_o = sizeof(float);
  int float1_o = sizeof(float1);
  // CHECK: int float_p = sizeof(float);
  int float_p = sizeof(float);
  // CHECK: int float1_q = sizeof(float1_d);
  int float1_q = sizeof(float1_d);
  int *float1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> float1_e_buf_ct0 = dpct::get_buffer_and_offset(float1_e);
  // CHECK-NEXT:   size_t float1_e_offset_ct0 = float1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> float1_cast_buf_ct1 = dpct::get_buffer_and_offset((float *)float1_cast);
  // CHECK-NEXT:   size_t float1_cast_offset_ct1 = float1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto float1_e_acc_ct0 = float1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto float1_cast_acc_ct1 = float1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_float1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           float *float1_e_ct0 = (float *)(&float1_e_acc_ct0[0] + float1_e_offset_ct0);
  // CHECK-NEXT:           float *float1_cast_ct1 = (float *)(&float1_cast_acc_ct1[0] + float1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_float1(float1_e_ct0, float1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_float1<<<1,1>>>(float1_e, (float1 *)float1_cast);
  return 0;
}

// CHECK: void func3_float2(cl::sycl::float2 a, cl::sycl::float2 b, cl::sycl::float2 c) {
void func3_float2(float2 a, float2 b, float2 c) {
}
// CHECK: void func_float2(cl::sycl::float2 a) {
void func_float2(float2 a) {
}
// CHECK: void kernel_float2(cl::sycl::float2 *a, cl::sycl::float2 *b) {
__global__ void kernel_float2(float2 *a, float2 *b) {
}

int main_float2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::float2 float2_a;
  float2 float2_a;
  // CHECK: cl::sycl::float2 float2_b = cl::sycl::float2(1, 2);
  float2 float2_b = make_float2(1, 2);
  // CHECK: cl::sycl::float2 float2_c = cl::sycl::float2(float2_b);
  float2 float2_c = float2(float2_b);
  // CHECK: cl::sycl::float2 float2_d(float2_c);
  float2 float2_d(float2_c);
  // CHECK: func3_float2(float2_b, cl::sycl::float2(float2_b), (cl::sycl::float2)float2_b);
  func3_float2(float2_b, float2(float2_b), (float2)float2_b);
  // CHECK: cl::sycl::float2 *float2_e;
  float2 *float2_e;
  // CHECK: cl::sycl::float2 *float2_f;
  float2 *float2_f;
  // CHECK: float float2_g = static_cast<float>(float2_c.x());
  float float2_g = float2_c.x;
  // CHECK: float2_a.x() = static_cast<float>(float2_d.x());
  float2_a.x = float2_d.x;
  // CHECK: if (static_cast<float>(float2_b.x()) == static_cast<float>(float2_d.x())) {}
  if (float2_b.x == float2_d.x) {}
  // CHECK: cl::sycl::float2 float2_h[16];
  float2 float2_h[16];
  // CHECK: cl::sycl::float2 float2_i[32];
  float2 float2_i[32];
  // CHECK: if (static_cast<float>(float2_h[12].x()) == static_cast<float>(float2_i[12].x())) {}
  if (float2_h[12].x == float2_i[12].x) {}
  // CHECK: float2_f = (cl::sycl::float2 *)float2_i;
  float2_f = (float2 *)float2_i;
  // CHECK: float2_a = (cl::sycl::float2)float2_c;
  float2_a = (float2)float2_c;
  // CHECK: float2_b = cl::sycl::float2(float2_b);
  float2_b = float2(float2_b);
  // CHECK: cl::sycl::float2 float2_j, float2_k, float2_l, float2_m[16], *float2_n[32];
  float2 float2_j, float2_k, float2_l, float2_m[16], *float2_n[32];
  // CHECK: int float2_o = sizeof(cl::sycl::float2);
  int float2_o = sizeof(float2);
  // CHECK: int float_p = sizeof(float);
  int float_p = sizeof(float);
  // CHECK: int float2_q = sizeof(float2_d);
  int float2_q = sizeof(float2_d);
  int *float2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> float2_e_buf_ct0 = dpct::get_buffer_and_offset(float2_e);
  // CHECK-NEXT:   size_t float2_e_offset_ct0 = float2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> float2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::float2 *)float2_cast);
  // CHECK-NEXT:   size_t float2_cast_offset_ct1 = float2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto float2_e_acc_ct0 = float2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto float2_cast_acc_ct1 = float2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_float2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::float2 *float2_e_ct0 = (cl::sycl::float2 *)(&float2_e_acc_ct0[0] + float2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::float2 *float2_cast_ct1 = (cl::sycl::float2 *)(&float2_cast_acc_ct1[0] + float2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_float2(float2_e_ct0, float2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_float2<<<1,1>>>(float2_e, (float2 *)float2_cast);
  return 0;
}

// CHECK: void func3_float3(cl::sycl::float3 a, cl::sycl::float3 b, cl::sycl::float3 c) {
void func3_float3(float3 a, float3 b, float3 c) {
}
// CHECK: void func_float3(cl::sycl::float3 a) {
void func_float3(float3 a) {
}
// CHECK: void kernel_float3(cl::sycl::float3 *a, cl::sycl::float3 *b) {
__global__ void kernel_float3(float3 *a, float3 *b) {
}

int main_float3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::float3 float3_a;
  float3 float3_a;
  // CHECK: cl::sycl::float3 float3_b = cl::sycl::float3(1, 2, 3);
  float3 float3_b = make_float3(1, 2, 3);
  // CHECK: cl::sycl::float3 float3_c = cl::sycl::float3(float3_b);
  float3 float3_c = float3(float3_b);
  // CHECK: cl::sycl::float3 float3_d(float3_c);
  float3 float3_d(float3_c);
  // CHECK: func3_float3(float3_b, cl::sycl::float3(float3_b), (cl::sycl::float3)float3_b);
  func3_float3(float3_b, float3(float3_b), (float3)float3_b);
  // CHECK: cl::sycl::float3 *float3_e;
  float3 *float3_e;
  // CHECK: cl::sycl::float3 *float3_f;
  float3 *float3_f;
  // CHECK: float float3_g = static_cast<float>(float3_c.x());
  float float3_g = float3_c.x;
  // CHECK: float3_a.x() = static_cast<float>(float3_d.x());
  float3_a.x = float3_d.x;
  // CHECK: if (static_cast<float>(float3_b.x()) == static_cast<float>(float3_d.x())) {}
  if (float3_b.x == float3_d.x) {}
  // CHECK: cl::sycl::float3 float3_h[16];
  float3 float3_h[16];
  // CHECK: cl::sycl::float3 float3_i[32];
  float3 float3_i[32];
  // CHECK: if (static_cast<float>(float3_h[12].x()) == static_cast<float>(float3_i[12].x())) {}
  if (float3_h[12].x == float3_i[12].x) {}
  // CHECK: float3_f = (cl::sycl::float3 *)float3_i;
  float3_f = (float3 *)float3_i;
  // CHECK: float3_a = (cl::sycl::float3)float3_c;
  float3_a = (float3)float3_c;
  // CHECK: float3_b = cl::sycl::float3(float3_b);
  float3_b = float3(float3_b);
  // CHECK: cl::sycl::float3 float3_j, float3_k, float3_l, float3_m[16], *float3_n[32];
  float3 float3_j, float3_k, float3_l, float3_m[16], *float3_n[32];
  // CHECK: int float3_o = sizeof(cl::sycl::float3);
  int float3_o = sizeof(float3);
  // CHECK: int float_p = sizeof(float);
  int float_p = sizeof(float);
  // CHECK: int float3_q = sizeof(float3_d);
  int float3_q = sizeof(float3_d);
  int *float3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> float3_e_buf_ct0 = dpct::get_buffer_and_offset(float3_e);
  // CHECK-NEXT:   size_t float3_e_offset_ct0 = float3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> float3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::float3 *)float3_cast);
  // CHECK-NEXT:   size_t float3_cast_offset_ct1 = float3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto float3_e_acc_ct0 = float3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto float3_cast_acc_ct1 = float3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_float3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::float3 *float3_e_ct0 = (cl::sycl::float3 *)(&float3_e_acc_ct0[0] + float3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::float3 *float3_cast_ct1 = (cl::sycl::float3 *)(&float3_cast_acc_ct1[0] + float3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_float3(float3_e_ct0, float3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_float3<<<1,1>>>(float3_e, (float3 *)float3_cast);
  return 0;
}

// CHECK: void func3_float4(cl::sycl::float4 a, cl::sycl::float4 b, cl::sycl::float4 c) {
void func3_float4(float4 a, float4 b, float4 c) {
}
// CHECK: void func_float4(cl::sycl::float4 a) {
void func_float4(float4 a) {
}
// CHECK: void kernel_float4(cl::sycl::float4 *a, cl::sycl::float4 *b) {
__global__ void kernel_float4(float4 *a, float4 *b) {
}

int main_float4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::float4 float4_a;
  float4 float4_a;
  // CHECK: cl::sycl::float4 float4_b = cl::sycl::float4(1, 2, 3, 4);
  float4 float4_b = make_float4(1, 2, 3, 4);
  // CHECK: cl::sycl::float4 float4_c = cl::sycl::float4(float4_b);
  float4 float4_c = float4(float4_b);
  // CHECK: cl::sycl::float4 float4_d(float4_c);
  float4 float4_d(float4_c);
  // CHECK: func3_float4(float4_b, cl::sycl::float4(float4_b), (cl::sycl::float4)float4_b);
  func3_float4(float4_b, float4(float4_b), (float4)float4_b);
  // CHECK: cl::sycl::float4 *float4_e;
  float4 *float4_e;
  // CHECK: cl::sycl::float4 *float4_f;
  float4 *float4_f;
  // CHECK: float float4_g = static_cast<float>(float4_c.x());
  float float4_g = float4_c.x;
  // CHECK: float4_a.x() = static_cast<float>(float4_d.x());
  float4_a.x = float4_d.x;
  // CHECK: if (static_cast<float>(float4_b.x()) == static_cast<float>(float4_d.x())) {}
  if (float4_b.x == float4_d.x) {}
  // CHECK: cl::sycl::float4 float4_h[16];
  float4 float4_h[16];
  // CHECK: cl::sycl::float4 float4_i[32];
  float4 float4_i[32];
  // CHECK: if (static_cast<float>(float4_h[12].x()) == static_cast<float>(float4_i[12].x())) {}
  if (float4_h[12].x == float4_i[12].x) {}
  // CHECK: float4_f = (cl::sycl::float4 *)float4_i;
  float4_f = (float4 *)float4_i;
  // CHECK: float4_a = (cl::sycl::float4)float4_c;
  float4_a = (float4)float4_c;
  // CHECK: float4_b = cl::sycl::float4(float4_b);
  float4_b = float4(float4_b);
  // CHECK: cl::sycl::float4 float4_j, float4_k, float4_l, float4_m[16], *float4_n[32];
  float4 float4_j, float4_k, float4_l, float4_m[16], *float4_n[32];
  // CHECK: int float4_o = sizeof(cl::sycl::float4);
  int float4_o = sizeof(float4);
  // CHECK: int float_p = sizeof(float);
  int float_p = sizeof(float);
  // CHECK: int float4_q = sizeof(float4_d);
  int float4_q = sizeof(float4_d);
  int *float4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> float4_e_buf_ct0 = dpct::get_buffer_and_offset(float4_e);
  // CHECK-NEXT:   size_t float4_e_offset_ct0 = float4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> float4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::float4 *)float4_cast);
  // CHECK-NEXT:   size_t float4_cast_offset_ct1 = float4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto float4_e_acc_ct0 = float4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto float4_cast_acc_ct1 = float4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_float4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::float4 *float4_e_ct0 = (cl::sycl::float4 *)(&float4_e_acc_ct0[0] + float4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::float4 *float4_cast_ct1 = (cl::sycl::float4 *)(&float4_cast_acc_ct1[0] + float4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_float4(float4_e_ct0, float4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_float4<<<1,1>>>(float4_e, (float4 *)float4_cast);
  return 0;
}

// CHECK: void func3_int1(int a, int b, int c) {
void func3_int1(int1 a, int1 b, int1 c) {
}
// CHECK: void func_int1(int a) {
void func_int1(int1 a) {
}
// CHECK: void kernel_int1(int *a, int *b) {
__global__ void kernel_int1(int1 *a, int1 *b) {
}

int main_int1() {
  // range default constructor does the right thing.
  // CHECK: int int1_a;
  int1 int1_a;
  // CHECK: int int1_b = int(1);
  int1 int1_b = make_int1(1);
  // CHECK: int int1_c = int(int1_b);
  int1 int1_c = int1(int1_b);
  // CHECK: int int1_d(int1_c);
  int1 int1_d(int1_c);
  // CHECK: func3_int1(int1_b, int(int1_b), (int)int1_b);
  func3_int1(int1_b, int1(int1_b), (int1)int1_b);
  // CHECK: int *int1_e;
  int1 *int1_e;
  // CHECK: int *int1_f;
  int1 *int1_f;
  // CHECK: int int1_g = static_cast<int>(int1_c);
  int int1_g = int1_c.x;
  // CHECK: int1_a = static_cast<int>(int1_d);
  int1_a.x = int1_d.x;
  // CHECK: if (static_cast<int>(int1_b) == static_cast<int>(int1_d)) {}
  if (int1_b.x == int1_d.x) {}
  // CHECK: int int1_h[16];
  int1 int1_h[16];
  // CHECK: int int1_i[32];
  int1 int1_i[32];
  // CHECK: if (static_cast<int>(int1_h[12]) == static_cast<int>(int1_i[12])) {}
  if (int1_h[12].x == int1_i[12].x) {}
  // CHECK: int1_f = (int *)int1_i;
  int1_f = (int1 *)int1_i;
  // CHECK: int1_a = (int)int1_c;
  int1_a = (int1)int1_c;
  // CHECK: int1_b = int(int1_b);
  int1_b = int1(int1_b);
  // CHECK: int int1_j, int1_k, int1_l, int1_m[16], *int1_n[32];
  int1 int1_j, int1_k, int1_l, int1_m[16], *int1_n[32];
  // CHECK: int int1_o = sizeof(int);
  int int1_o = sizeof(int1);
  // CHECK: int int_p = sizeof(int);
  int int_p = sizeof(int);
  // CHECK: int int1_q = sizeof(int1_d);
  int int1_q = sizeof(int1_d);
  int *int1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> int1_e_buf_ct0 = dpct::get_buffer_and_offset(int1_e);
  // CHECK-NEXT:   size_t int1_e_offset_ct0 = int1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> int1_cast_buf_ct1 = dpct::get_buffer_and_offset((int *)int1_cast);
  // CHECK-NEXT:   size_t int1_cast_offset_ct1 = int1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto int1_e_acc_ct0 = int1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto int1_cast_acc_ct1 = int1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_int1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           int *int1_e_ct0 = (int *)(&int1_e_acc_ct0[0] + int1_e_offset_ct0);
  // CHECK-NEXT:           int *int1_cast_ct1 = (int *)(&int1_cast_acc_ct1[0] + int1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_int1(int1_e_ct0, int1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_int1<<<1,1>>>(int1_e, (int1 *)int1_cast);
  return 0;
}

// CHECK: void func3_int2(cl::sycl::int2 a, cl::sycl::int2 b, cl::sycl::int2 c) {
void func3_int2(int2 a, int2 b, int2 c) {
}
// CHECK: void func_int2(cl::sycl::int2 a) {
void func_int2(int2 a) {
}
// CHECK: void kernel_int2(cl::sycl::int2 *a, cl::sycl::int2 *b) {
__global__ void kernel_int2(int2 *a, int2 *b) {
}

int main_int2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::int2 int2_a;
  int2 int2_a;
  // CHECK: cl::sycl::int2 int2_b = cl::sycl::int2(1, 2);
  int2 int2_b = make_int2(1, 2);
  // CHECK: cl::sycl::int2 int2_c = cl::sycl::int2(int2_b);
  int2 int2_c = int2(int2_b);
  // CHECK: cl::sycl::int2 int2_d(int2_c);
  int2 int2_d(int2_c);
  // CHECK: func3_int2(int2_b, cl::sycl::int2(int2_b), (cl::sycl::int2)int2_b);
  func3_int2(int2_b, int2(int2_b), (int2)int2_b);
  // CHECK: cl::sycl::int2 *int2_e;
  int2 *int2_e;
  // CHECK: cl::sycl::int2 *int2_f;
  int2 *int2_f;
  // CHECK: int int2_g = static_cast<int>(int2_c.x());
  int int2_g = int2_c.x;
  // CHECK: int2_a.x() = static_cast<int>(int2_d.x());
  int2_a.x = int2_d.x;
  // CHECK: if (static_cast<int>(int2_b.x()) == static_cast<int>(int2_d.x())) {}
  if (int2_b.x == int2_d.x) {}
  // CHECK: cl::sycl::int2 int2_h[16];
  int2 int2_h[16];
  // CHECK: cl::sycl::int2 int2_i[32];
  int2 int2_i[32];
  // CHECK: if (static_cast<int>(int2_h[12].x()) == static_cast<int>(int2_i[12].x())) {}
  if (int2_h[12].x == int2_i[12].x) {}
  // CHECK: int2_f = (cl::sycl::int2 *)int2_i;
  int2_f = (int2 *)int2_i;
  // CHECK: int2_a = (cl::sycl::int2)int2_c;
  int2_a = (int2)int2_c;
  // CHECK: int2_b = cl::sycl::int2(int2_b);
  int2_b = int2(int2_b);
  // CHECK: cl::sycl::int2 int2_j, int2_k, int2_l, int2_m[16], *int2_n[32];
  int2 int2_j, int2_k, int2_l, int2_m[16], *int2_n[32];
  // CHECK: int int2_o = sizeof(cl::sycl::int2);
  int int2_o = sizeof(int2);
  // CHECK: int int_p = sizeof(int);
  int int_p = sizeof(int);
  // CHECK: int int2_q = sizeof(int2_d);
  int int2_q = sizeof(int2_d);
  int *int2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> int2_e_buf_ct0 = dpct::get_buffer_and_offset(int2_e);
  // CHECK-NEXT:   size_t int2_e_offset_ct0 = int2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> int2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::int2 *)int2_cast);
  // CHECK-NEXT:   size_t int2_cast_offset_ct1 = int2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto int2_e_acc_ct0 = int2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto int2_cast_acc_ct1 = int2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_int2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::int2 *int2_e_ct0 = (cl::sycl::int2 *)(&int2_e_acc_ct0[0] + int2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::int2 *int2_cast_ct1 = (cl::sycl::int2 *)(&int2_cast_acc_ct1[0] + int2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_int2(int2_e_ct0, int2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_int2<<<1,1>>>(int2_e, (int2 *)int2_cast);
  return 0;
}

// CHECK: void func3_int3(cl::sycl::int3 a, cl::sycl::int3 b, cl::sycl::int3 c) {
void func3_int3(int3 a, int3 b, int3 c) {
}
// CHECK: void func_int3(cl::sycl::int3 a) {
void func_int3(int3 a) {
}
// CHECK: void kernel_int3(cl::sycl::int3 *a, cl::sycl::int3 *b) {
__global__ void kernel_int3(int3 *a, int3 *b) {
}

int main_int3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::int3 int3_a;
  int3 int3_a;
  // CHECK: cl::sycl::int3 int3_b = cl::sycl::int3(1, 2, 3);
  int3 int3_b = make_int3(1, 2, 3);
  // CHECK: cl::sycl::int3 int3_c = cl::sycl::int3(int3_b);
  int3 int3_c = int3(int3_b);
  // CHECK: cl::sycl::int3 int3_d(int3_c);
  int3 int3_d(int3_c);
  // CHECK: func3_int3(int3_b, cl::sycl::int3(int3_b), (cl::sycl::int3)int3_b);
  func3_int3(int3_b, int3(int3_b), (int3)int3_b);
  // CHECK: cl::sycl::int3 *int3_e;
  int3 *int3_e;
  // CHECK: cl::sycl::int3 *int3_f;
  int3 *int3_f;
  // CHECK: int int3_g = static_cast<int>(int3_c.x());
  int int3_g = int3_c.x;
  // CHECK: int3_a.x() = static_cast<int>(int3_d.x());
  int3_a.x = int3_d.x;
  // CHECK: if (static_cast<int>(int3_b.x()) == static_cast<int>(int3_d.x())) {}
  if (int3_b.x == int3_d.x) {}
  // CHECK: cl::sycl::int3 int3_h[16];
  int3 int3_h[16];
  // CHECK: cl::sycl::int3 int3_i[32];
  int3 int3_i[32];
  // CHECK: if (static_cast<int>(int3_h[12].x()) == static_cast<int>(int3_i[12].x())) {}
  if (int3_h[12].x == int3_i[12].x) {}
  // CHECK: int3_f = (cl::sycl::int3 *)int3_i;
  int3_f = (int3 *)int3_i;
  // CHECK: int3_a = (cl::sycl::int3)int3_c;
  int3_a = (int3)int3_c;
  // CHECK: int3_b = cl::sycl::int3(int3_b);
  int3_b = int3(int3_b);
  // CHECK: cl::sycl::int3 int3_j, int3_k, int3_l, int3_m[16], *int3_n[32];
  int3 int3_j, int3_k, int3_l, int3_m[16], *int3_n[32];
  // CHECK: int int3_o = sizeof(cl::sycl::int3);
  int int3_o = sizeof(int3);
  // CHECK: int int_p = sizeof(int);
  int int_p = sizeof(int);
  // CHECK: int int3_q = sizeof(int3_d);
  int int3_q = sizeof(int3_d);
  int *int3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> int3_e_buf_ct0 = dpct::get_buffer_and_offset(int3_e);
  // CHECK-NEXT:   size_t int3_e_offset_ct0 = int3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> int3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::int3 *)int3_cast);
  // CHECK-NEXT:   size_t int3_cast_offset_ct1 = int3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto int3_e_acc_ct0 = int3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto int3_cast_acc_ct1 = int3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_int3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::int3 *int3_e_ct0 = (cl::sycl::int3 *)(&int3_e_acc_ct0[0] + int3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::int3 *int3_cast_ct1 = (cl::sycl::int3 *)(&int3_cast_acc_ct1[0] + int3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_int3(int3_e_ct0, int3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_int3<<<1,1>>>(int3_e, (int3 *)int3_cast);
  return 0;
}

// CHECK: void func3_int4(cl::sycl::int4 a, cl::sycl::int4 b, cl::sycl::int4 c) {
void func3_int4(int4 a, int4 b, int4 c) {
}
// CHECK: void func_int4(cl::sycl::int4 a) {
void func_int4(int4 a) {
}
// CHECK: void kernel_int4(cl::sycl::int4 *a, cl::sycl::int4 *b) {
__global__ void kernel_int4(int4 *a, int4 *b) {
}

int main_int4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::int4 int4_a;
  int4 int4_a;
  // CHECK: cl::sycl::int4 int4_b = cl::sycl::int4(1, 2, 3, 4);
  int4 int4_b = make_int4(1, 2, 3, 4);
  // CHECK: cl::sycl::int4 int4_c = cl::sycl::int4(int4_b);
  int4 int4_c = int4(int4_b);
  // CHECK: cl::sycl::int4 int4_d(int4_c);
  int4 int4_d(int4_c);
  // CHECK: func3_int4(int4_b, cl::sycl::int4(int4_b), (cl::sycl::int4)int4_b);
  func3_int4(int4_b, int4(int4_b), (int4)int4_b);
  // CHECK: cl::sycl::int4 *int4_e;
  int4 *int4_e;
  // CHECK: cl::sycl::int4 *int4_f;
  int4 *int4_f;
  // CHECK: int int4_g = static_cast<int>(int4_c.x());
  int int4_g = int4_c.x;
  // CHECK: int4_a.x() = static_cast<int>(int4_d.x());
  int4_a.x = int4_d.x;
  // CHECK: if (static_cast<int>(int4_b.x()) == static_cast<int>(int4_d.x())) {}
  if (int4_b.x == int4_d.x) {}
  // CHECK: cl::sycl::int4 int4_h[16];
  int4 int4_h[16];
  // CHECK: cl::sycl::int4 int4_i[32];
  int4 int4_i[32];
  // CHECK: if (static_cast<int>(int4_h[12].x()) == static_cast<int>(int4_i[12].x())) {}
  if (int4_h[12].x == int4_i[12].x) {}
  // CHECK: int4_f = (cl::sycl::int4 *)int4_i;
  int4_f = (int4 *)int4_i;
  // CHECK: int4_a = (cl::sycl::int4)int4_c;
  int4_a = (int4)int4_c;
  // CHECK: int4_b = cl::sycl::int4(int4_b);
  int4_b = int4(int4_b);
  // CHECK: cl::sycl::int4 int4_j, int4_k, int4_l, int4_m[16], *int4_n[32];
  int4 int4_j, int4_k, int4_l, int4_m[16], *int4_n[32];
  // CHECK: int int4_o = sizeof(cl::sycl::int4);
  int int4_o = sizeof(int4);
  // CHECK: int int_p = sizeof(int);
  int int_p = sizeof(int);
  // CHECK: int int4_q = sizeof(int4_d);
  int int4_q = sizeof(int4_d);
  int *int4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> int4_e_buf_ct0 = dpct::get_buffer_and_offset(int4_e);
  // CHECK-NEXT:   size_t int4_e_offset_ct0 = int4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> int4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::int4 *)int4_cast);
  // CHECK-NEXT:   size_t int4_cast_offset_ct1 = int4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto int4_e_acc_ct0 = int4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto int4_cast_acc_ct1 = int4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_int4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::int4 *int4_e_ct0 = (cl::sycl::int4 *)(&int4_e_acc_ct0[0] + int4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::int4 *int4_cast_ct1 = (cl::sycl::int4 *)(&int4_cast_acc_ct1[0] + int4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_int4(int4_e_ct0, int4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_int4<<<1,1>>>(int4_e, (int4 *)int4_cast);
  return 0;
}

// CHECK: void func3_long1(long a, long b, long c) {
void func3_long1(long1 a, long1 b, long1 c) {
}
// CHECK: void func_long1(long a) {
void func_long1(long1 a) {
}
// CHECK: void kernel_long1(long *a, long *b) {
__global__ void kernel_long1(long1 *a, long1 *b) {
}

int main_long1() {
  // range default constructor does the right thing.
  // CHECK: long long1_a;
  long1 long1_a;
  // CHECK: long long1_b = long(1);
  long1 long1_b = make_long1(1);
  // CHECK: long long1_c = long(long1_b);
  long1 long1_c = long1(long1_b);
  // CHECK: long long1_d(long1_c);
  long1 long1_d(long1_c);
  // CHECK: func3_long1(long1_b, long(long1_b), (long)long1_b);
  func3_long1(long1_b, long1(long1_b), (long1)long1_b);
  // CHECK: long *long1_e;
  long1 *long1_e;
  // CHECK: long *long1_f;
  long1 *long1_f;
  // CHECK: long long1_g = static_cast<long>(long1_c);
  long long1_g = long1_c.x;
  // CHECK: long1_a = static_cast<long>(long1_d);
  long1_a.x = long1_d.x;
  // CHECK: if (static_cast<long>(long1_b) == static_cast<long>(long1_d)) {}
  if (long1_b.x == long1_d.x) {}
  // CHECK: long long1_h[16];
  long1 long1_h[16];
  // CHECK: long long1_i[32];
  long1 long1_i[32];
  // CHECK: if (static_cast<long>(long1_h[12]) == static_cast<long>(long1_i[12])) {}
  if (long1_h[12].x == long1_i[12].x) {}
  // CHECK: long1_f = (long *)long1_i;
  long1_f = (long1 *)long1_i;
  // CHECK: long1_a = (long)long1_c;
  long1_a = (long1)long1_c;
  // CHECK: long1_b = long(long1_b);
  long1_b = long1(long1_b);
  // CHECK: long long1_j, long1_k, long1_l, long1_m[16], *long1_n[32];
  long1 long1_j, long1_k, long1_l, long1_m[16], *long1_n[32];
  // CHECK: int long1_o = sizeof(long);
  int long1_o = sizeof(long1);
  // CHECK: int long_p = sizeof(long);
  int long_p = sizeof(long);
  // CHECK: int long1_q = sizeof(long1_d);
  int long1_q = sizeof(long1_d);
  int *long1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> long1_e_buf_ct0 = dpct::get_buffer_and_offset(long1_e);
  // CHECK-NEXT:   size_t long1_e_offset_ct0 = long1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> long1_cast_buf_ct1 = dpct::get_buffer_and_offset((long *)long1_cast);
  // CHECK-NEXT:   size_t long1_cast_offset_ct1 = long1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto long1_e_acc_ct0 = long1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto long1_cast_acc_ct1 = long1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_long1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           long *long1_e_ct0 = (long *)(&long1_e_acc_ct0[0] + long1_e_offset_ct0);
  // CHECK-NEXT:           long *long1_cast_ct1 = (long *)(&long1_cast_acc_ct1[0] + long1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_long1(long1_e_ct0, long1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_long1<<<1,1>>>(long1_e, (long1 *)long1_cast);
  return 0;
}

// CHECK: void func3_long2(cl::sycl::long2 a, cl::sycl::long2 b, cl::sycl::long2 c) {
void func3_long2(long2 a, long2 b, long2 c) {
}
// CHECK: void func_long2(cl::sycl::long2 a) {
void func_long2(long2 a) {
}
// CHECK: void kernel_long2(cl::sycl::long2 *a, cl::sycl::long2 *b) {
__global__ void kernel_long2(long2 *a, long2 *b) {
}

int main_long2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::long2 long2_a;
  long2 long2_a;
  // CHECK: cl::sycl::long2 long2_b = cl::sycl::long2(1, 2);
  long2 long2_b = make_long2(1, 2);
  // CHECK: cl::sycl::long2 long2_c = cl::sycl::long2(long2_b);
  long2 long2_c = long2(long2_b);
  // CHECK: cl::sycl::long2 long2_d(long2_c);
  long2 long2_d(long2_c);
  // CHECK: func3_long2(long2_b, cl::sycl::long2(long2_b), (cl::sycl::long2)long2_b);
  func3_long2(long2_b, long2(long2_b), (long2)long2_b);
  // CHECK: cl::sycl::long2 *long2_e;
  long2 *long2_e;
  // CHECK: cl::sycl::long2 *long2_f;
  long2 *long2_f;
  // CHECK: long long2_g = static_cast<long>(long2_c.x());
  long long2_g = long2_c.x;
  // CHECK: long2_a.x() = static_cast<long>(long2_d.x());
  long2_a.x = long2_d.x;
  // CHECK: if (static_cast<long>(long2_b.x()) == static_cast<long>(long2_d.x())) {}
  if (long2_b.x == long2_d.x) {}
  // CHECK: cl::sycl::long2 long2_h[16];
  long2 long2_h[16];
  // CHECK: cl::sycl::long2 long2_i[32];
  long2 long2_i[32];
  // CHECK: if (static_cast<long>(long2_h[12].x()) == static_cast<long>(long2_i[12].x())) {}
  if (long2_h[12].x == long2_i[12].x) {}
  // CHECK: long2_f = (cl::sycl::long2 *)long2_i;
  long2_f = (long2 *)long2_i;
  // CHECK: long2_a = (cl::sycl::long2)long2_c;
  long2_a = (long2)long2_c;
  // CHECK: long2_b = cl::sycl::long2(long2_b);
  long2_b = long2(long2_b);
  // CHECK: cl::sycl::long2 long2_j, long2_k, long2_l, long2_m[16], *long2_n[32];
  long2 long2_j, long2_k, long2_l, long2_m[16], *long2_n[32];
  // CHECK: int long2_o = sizeof(cl::sycl::long2);
  int long2_o = sizeof(long2);
  // CHECK: int long_p = sizeof(long);
  int long_p = sizeof(long);
  // CHECK: int long2_q = sizeof(long2_d);
  int long2_q = sizeof(long2_d);
  int *long2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> long2_e_buf_ct0 = dpct::get_buffer_and_offset(long2_e);
  // CHECK-NEXT:   size_t long2_e_offset_ct0 = long2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> long2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::long2 *)long2_cast);
  // CHECK-NEXT:   size_t long2_cast_offset_ct1 = long2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto long2_e_acc_ct0 = long2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto long2_cast_acc_ct1 = long2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_long2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::long2 *long2_e_ct0 = (cl::sycl::long2 *)(&long2_e_acc_ct0[0] + long2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::long2 *long2_cast_ct1 = (cl::sycl::long2 *)(&long2_cast_acc_ct1[0] + long2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_long2(long2_e_ct0, long2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_long2<<<1,1>>>(long2_e, (long2 *)long2_cast);
  return 0;
}

// CHECK: void func3_long3(cl::sycl::long3 a, cl::sycl::long3 b, cl::sycl::long3 c) {
void func3_long3(long3 a, long3 b, long3 c) {
}
// CHECK: void func_long3(cl::sycl::long3 a) {
void func_long3(long3 a) {
}
// CHECK: void kernel_long3(cl::sycl::long3 *a, cl::sycl::long3 *b) {
__global__ void kernel_long3(long3 *a, long3 *b) {
}

int main_long3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::long3 long3_a;
  long3 long3_a;
  // CHECK: cl::sycl::long3 long3_b = cl::sycl::long3(1, 2, 3);
  long3 long3_b = make_long3(1, 2, 3);
  // CHECK: cl::sycl::long3 long3_c = cl::sycl::long3(long3_b);
  long3 long3_c = long3(long3_b);
  // CHECK: cl::sycl::long3 long3_d(long3_c);
  long3 long3_d(long3_c);
  // CHECK: func3_long3(long3_b, cl::sycl::long3(long3_b), (cl::sycl::long3)long3_b);
  func3_long3(long3_b, long3(long3_b), (long3)long3_b);
  // CHECK: cl::sycl::long3 *long3_e;
  long3 *long3_e;
  // CHECK: cl::sycl::long3 *long3_f;
  long3 *long3_f;
  // CHECK: long long3_g = static_cast<long>(long3_c.x());
  long long3_g = long3_c.x;
  // CHECK: long3_a.x() = static_cast<long>(long3_d.x());
  long3_a.x = long3_d.x;
  // CHECK: if (static_cast<long>(long3_b.x()) == static_cast<long>(long3_d.x())) {}
  if (long3_b.x == long3_d.x) {}
  // CHECK: cl::sycl::long3 long3_h[16];
  long3 long3_h[16];
  // CHECK: cl::sycl::long3 long3_i[32];
  long3 long3_i[32];
  // CHECK: if (static_cast<long>(long3_h[12].x()) == static_cast<long>(long3_i[12].x())) {}
  if (long3_h[12].x == long3_i[12].x) {}
  // CHECK: long3_f = (cl::sycl::long3 *)long3_i;
  long3_f = (long3 *)long3_i;
  // CHECK: long3_a = (cl::sycl::long3)long3_c;
  long3_a = (long3)long3_c;
  // CHECK: long3_b = cl::sycl::long3(long3_b);
  long3_b = long3(long3_b);
  // CHECK: cl::sycl::long3 long3_j, long3_k, long3_l, long3_m[16], *long3_n[32];
  long3 long3_j, long3_k, long3_l, long3_m[16], *long3_n[32];
  // CHECK: int long3_o = sizeof(cl::sycl::long3);
  int long3_o = sizeof(long3);
  // CHECK: int long_p = sizeof(long);
  int long_p = sizeof(long);
  // CHECK: int long3_q = sizeof(long3_d);
  int long3_q = sizeof(long3_d);
  int *long3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> long3_e_buf_ct0 = dpct::get_buffer_and_offset(long3_e);
  // CHECK-NEXT:   size_t long3_e_offset_ct0 = long3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> long3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::long3 *)long3_cast);
  // CHECK-NEXT:   size_t long3_cast_offset_ct1 = long3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto long3_e_acc_ct0 = long3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto long3_cast_acc_ct1 = long3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_long3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::long3 *long3_e_ct0 = (cl::sycl::long3 *)(&long3_e_acc_ct0[0] + long3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::long3 *long3_cast_ct1 = (cl::sycl::long3 *)(&long3_cast_acc_ct1[0] + long3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_long3(long3_e_ct0, long3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_long3<<<1,1>>>(long3_e, (long3 *)long3_cast);
  return 0;
}

// CHECK: void func3_long4(cl::sycl::long4 a, cl::sycl::long4 b, cl::sycl::long4 c) {
void func3_long4(long4 a, long4 b, long4 c) {
}
// CHECK: void func_long4(cl::sycl::long4 a) {
void func_long4(long4 a) {
}
// CHECK: void kernel_long4(cl::sycl::long4 *a, cl::sycl::long4 *b) {
__global__ void kernel_long4(long4 *a, long4 *b) {
}

int main_long4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::long4 long4_a;
  long4 long4_a;
  // CHECK: cl::sycl::long4 long4_b = cl::sycl::long4(1, 2, 3, 4);
  long4 long4_b = make_long4(1, 2, 3, 4);
  // CHECK: cl::sycl::long4 long4_c = cl::sycl::long4(long4_b);
  long4 long4_c = long4(long4_b);
  // CHECK: cl::sycl::long4 long4_d(long4_c);
  long4 long4_d(long4_c);
  // CHECK: func3_long4(long4_b, cl::sycl::long4(long4_b), (cl::sycl::long4)long4_b);
  func3_long4(long4_b, long4(long4_b), (long4)long4_b);
  // CHECK: cl::sycl::long4 *long4_e;
  long4 *long4_e;
  // CHECK: cl::sycl::long4 *long4_f;
  long4 *long4_f;
  // CHECK: long long4_g = static_cast<long>(long4_c.x());
  long long4_g = long4_c.x;
  // CHECK: long4_a.x() = static_cast<long>(long4_d.x());
  long4_a.x = long4_d.x;
  // CHECK: if (static_cast<long>(long4_b.x()) == static_cast<long>(long4_d.x())) {}
  if (long4_b.x == long4_d.x) {}
  // CHECK: cl::sycl::long4 long4_h[16];
  long4 long4_h[16];
  // CHECK: cl::sycl::long4 long4_i[32];
  long4 long4_i[32];
  // CHECK: if (static_cast<long>(long4_h[12].x()) == static_cast<long>(long4_i[12].x())) {}
  if (long4_h[12].x == long4_i[12].x) {}
  // CHECK: long4_f = (cl::sycl::long4 *)long4_i;
  long4_f = (long4 *)long4_i;
  // CHECK: long4_a = (cl::sycl::long4)long4_c;
  long4_a = (long4)long4_c;
  // CHECK: long4_b = cl::sycl::long4(long4_b);
  long4_b = long4(long4_b);
  // CHECK: cl::sycl::long4 long4_j, long4_k, long4_l, long4_m[16], *long4_n[32];
  long4 long4_j, long4_k, long4_l, long4_m[16], *long4_n[32];
  // CHECK: int long4_o = sizeof(cl::sycl::long4);
  int long4_o = sizeof(long4);
  // CHECK: int long_p = sizeof(long);
  int long_p = sizeof(long);
  // CHECK: int long4_q = sizeof(long4_d);
  int long4_q = sizeof(long4_d);
  int *long4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> long4_e_buf_ct0 = dpct::get_buffer_and_offset(long4_e);
  // CHECK-NEXT:   size_t long4_e_offset_ct0 = long4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> long4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::long4 *)long4_cast);
  // CHECK-NEXT:   size_t long4_cast_offset_ct1 = long4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto long4_e_acc_ct0 = long4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto long4_cast_acc_ct1 = long4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_long4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::long4 *long4_e_ct0 = (cl::sycl::long4 *)(&long4_e_acc_ct0[0] + long4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::long4 *long4_cast_ct1 = (cl::sycl::long4 *)(&long4_cast_acc_ct1[0] + long4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_long4(long4_e_ct0, long4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_long4<<<1,1>>>(long4_e, (long4 *)long4_cast);
  return 0;
}

// CHECK: void func3_longlong1(long long a, long long b, long long c) {
void func3_longlong1(longlong1 a, longlong1 b, longlong1 c) {
}
// CHECK: void func_longlong1(long long a) {
void func_longlong1(longlong1 a) {
}
// CHECK: void kernel_longlong1(long long *a, long long *b) {
__global__ void kernel_longlong1(longlong1 *a, longlong1 *b) {
}

int main_longlong1() {
  // range default constructor does the right thing.
  // CHECK: long long longlong1_a;
  longlong1 longlong1_a;
  // CHECK: long long longlong1_b = long long(1);
  longlong1 longlong1_b = make_longlong1(1);
  // CHECK: long long longlong1_c = long long(longlong1_b);
  longlong1 longlong1_c = longlong1(longlong1_b);
  // CHECK: long long longlong1_d(longlong1_c);
  longlong1 longlong1_d(longlong1_c);
  // CHECK: func3_longlong1(longlong1_b, long long(longlong1_b), (long long)longlong1_b);
  func3_longlong1(longlong1_b, longlong1(longlong1_b), (longlong1)longlong1_b);
  // CHECK: long long *longlong1_e;
  longlong1 *longlong1_e;
  // CHECK: long long *longlong1_f;
  longlong1 *longlong1_f;
  // CHECK: long long longlong1_g = static_cast<long long>(longlong1_c);
  long long longlong1_g = longlong1_c.x;
  // CHECK: longlong1_a = static_cast<long long>(longlong1_d);
  longlong1_a.x = longlong1_d.x;
  // CHECK: if (static_cast<long long>(longlong1_b) == static_cast<long long>(longlong1_d)) {}
  if (longlong1_b.x == longlong1_d.x) {}
  // CHECK: long long longlong1_h[16];
  longlong1 longlong1_h[16];
  // CHECK: long long longlong1_i[32];
  longlong1 longlong1_i[32];
  // CHECK: if (static_cast<long long>(longlong1_h[12]) == static_cast<long long>(longlong1_i[12])) {}
  if (longlong1_h[12].x == longlong1_i[12].x) {}
  // CHECK: longlong1_f = (long long *)longlong1_i;
  longlong1_f = (longlong1 *)longlong1_i;
  // CHECK: longlong1_a = (long long)longlong1_c;
  longlong1_a = (longlong1)longlong1_c;
  // CHECK: longlong1_b = long long(longlong1_b);
  longlong1_b = longlong1(longlong1_b);
  // CHECK: long long longlong1_j, longlong1_k, longlong1_l, longlong1_m[16], *longlong1_n[32];
  longlong1 longlong1_j, longlong1_k, longlong1_l, longlong1_m[16], *longlong1_n[32];
  // CHECK: int longlong1_o = sizeof(long long);
  int longlong1_o = sizeof(longlong1);
  // CHECK: int long long_p = sizeof(long long);
  int long long_p = sizeof(long long);
  // CHECK: int longlong1_q = sizeof(longlong1_d);
  int longlong1_q = sizeof(longlong1_d);
  int *longlong1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> longlong1_e_buf_ct0 = dpct::get_buffer_and_offset(longlong1_e);
  // CHECK-NEXT:   size_t longlong1_e_offset_ct0 = longlong1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> longlong1_cast_buf_ct1 = dpct::get_buffer_and_offset((long long *)longlong1_cast);
  // CHECK-NEXT:   size_t longlong1_cast_offset_ct1 = longlong1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto longlong1_e_acc_ct0 = longlong1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto longlong1_cast_acc_ct1 = longlong1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_longlong1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           long long *longlong1_e_ct0 = (long long *)(&longlong1_e_acc_ct0[0] + longlong1_e_offset_ct0);
  // CHECK-NEXT:           long long *longlong1_cast_ct1 = (long long *)(&longlong1_cast_acc_ct1[0] + longlong1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_longlong1(longlong1_e_ct0, longlong1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_longlong1<<<1,1>>>(longlong1_e, (longlong1 *)longlong1_cast);
  return 0;
}

// CHECK: void func3_longlong2(cl::sycl::longlong2 a, cl::sycl::longlong2 b, cl::sycl::longlong2 c) {
void func3_longlong2(longlong2 a, longlong2 b, longlong2 c) {
}
// CHECK: void func_longlong2(cl::sycl::longlong2 a) {
void func_longlong2(longlong2 a) {
}
// CHECK: void kernel_longlong2(cl::sycl::longlong2 *a, cl::sycl::longlong2 *b) {
__global__ void kernel_longlong2(longlong2 *a, longlong2 *b) {
}

int main_longlong2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::longlong2 longlong2_a;
  longlong2 longlong2_a;
  // CHECK: cl::sycl::longlong2 longlong2_b = cl::sycl::longlong2(1, 2);
  longlong2 longlong2_b = make_longlong2(1, 2);
  // CHECK: cl::sycl::longlong2 longlong2_c = cl::sycl::longlong2(longlong2_b);
  longlong2 longlong2_c = longlong2(longlong2_b);
  // CHECK: cl::sycl::longlong2 longlong2_d(longlong2_c);
  longlong2 longlong2_d(longlong2_c);
  // CHECK: func3_longlong2(longlong2_b, cl::sycl::longlong2(longlong2_b), (cl::sycl::longlong2)longlong2_b);
  func3_longlong2(longlong2_b, longlong2(longlong2_b), (longlong2)longlong2_b);
  // CHECK: cl::sycl::longlong2 *longlong2_e;
  longlong2 *longlong2_e;
  // CHECK: cl::sycl::longlong2 *longlong2_f;
  longlong2 *longlong2_f;
  // CHECK: long long longlong2_g = static_cast<long long>(longlong2_c.x());
  long long longlong2_g = longlong2_c.x;
  // CHECK: longlong2_a.x() = static_cast<long long>(longlong2_d.x());
  longlong2_a.x = longlong2_d.x;
  // CHECK: if (static_cast<long long>(longlong2_b.x()) == static_cast<long long>(longlong2_d.x())) {}
  if (longlong2_b.x == longlong2_d.x) {}
  // CHECK: cl::sycl::longlong2 longlong2_h[16];
  longlong2 longlong2_h[16];
  // CHECK: cl::sycl::longlong2 longlong2_i[32];
  longlong2 longlong2_i[32];
  // CHECK: if (static_cast<long long>(longlong2_h[12].x()) == static_cast<long long>(longlong2_i[12].x())) {}
  if (longlong2_h[12].x == longlong2_i[12].x) {}
  // CHECK: longlong2_f = (cl::sycl::longlong2 *)longlong2_i;
  longlong2_f = (longlong2 *)longlong2_i;
  // CHECK: longlong2_a = (cl::sycl::longlong2)longlong2_c;
  longlong2_a = (longlong2)longlong2_c;
  // CHECK: longlong2_b = cl::sycl::longlong2(longlong2_b);
  longlong2_b = longlong2(longlong2_b);
  // CHECK: cl::sycl::longlong2 longlong2_j, longlong2_k, longlong2_l, longlong2_m[16], *longlong2_n[32];
  longlong2 longlong2_j, longlong2_k, longlong2_l, longlong2_m[16], *longlong2_n[32];
  // CHECK: int longlong2_o = sizeof(cl::sycl::longlong2);
  int longlong2_o = sizeof(longlong2);
  // CHECK: int long long_p = sizeof(long long);
  int long long_p = sizeof(long long);
  // CHECK: int longlong2_q = sizeof(longlong2_d);
  int longlong2_q = sizeof(longlong2_d);
  int *longlong2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> longlong2_e_buf_ct0 = dpct::get_buffer_and_offset(longlong2_e);
  // CHECK-NEXT:   size_t longlong2_e_offset_ct0 = longlong2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> longlong2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::longlong2 *)longlong2_cast);
  // CHECK-NEXT:   size_t longlong2_cast_offset_ct1 = longlong2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto longlong2_e_acc_ct0 = longlong2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto longlong2_cast_acc_ct1 = longlong2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_longlong2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::longlong2 *longlong2_e_ct0 = (cl::sycl::longlong2 *)(&longlong2_e_acc_ct0[0] + longlong2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::longlong2 *longlong2_cast_ct1 = (cl::sycl::longlong2 *)(&longlong2_cast_acc_ct1[0] + longlong2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_longlong2(longlong2_e_ct0, longlong2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_longlong2<<<1,1>>>(longlong2_e, (longlong2 *)longlong2_cast);
  return 0;
}

// CHECK: void func3_longlong3(cl::sycl::longlong3 a, cl::sycl::longlong3 b, cl::sycl::longlong3 c) {
void func3_longlong3(longlong3 a, longlong3 b, longlong3 c) {
}
// CHECK: void func_longlong3(cl::sycl::longlong3 a) {
void func_longlong3(longlong3 a) {
}
// CHECK: void kernel_longlong3(cl::sycl::longlong3 *a, cl::sycl::longlong3 *b) {
__global__ void kernel_longlong3(longlong3 *a, longlong3 *b) {
}

int main_longlong3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::longlong3 longlong3_a;
  longlong3 longlong3_a;
  // CHECK: cl::sycl::longlong3 longlong3_b = cl::sycl::longlong3(1, 2, 3);
  longlong3 longlong3_b = make_longlong3(1, 2, 3);
  // CHECK: cl::sycl::longlong3 longlong3_c = cl::sycl::longlong3(longlong3_b);
  longlong3 longlong3_c = longlong3(longlong3_b);
  // CHECK: cl::sycl::longlong3 longlong3_d(longlong3_c);
  longlong3 longlong3_d(longlong3_c);
  // CHECK: func3_longlong3(longlong3_b, cl::sycl::longlong3(longlong3_b), (cl::sycl::longlong3)longlong3_b);
  func3_longlong3(longlong3_b, longlong3(longlong3_b), (longlong3)longlong3_b);
  // CHECK: cl::sycl::longlong3 *longlong3_e;
  longlong3 *longlong3_e;
  // CHECK: cl::sycl::longlong3 *longlong3_f;
  longlong3 *longlong3_f;
  // CHECK: long long longlong3_g = static_cast<long long>(longlong3_c.x());
  long long longlong3_g = longlong3_c.x;
  // CHECK: longlong3_a.x() = static_cast<long long>(longlong3_d.x());
  longlong3_a.x = longlong3_d.x;
  // CHECK: if (static_cast<long long>(longlong3_b.x()) == static_cast<long long>(longlong3_d.x())) {}
  if (longlong3_b.x == longlong3_d.x) {}
  // CHECK: cl::sycl::longlong3 longlong3_h[16];
  longlong3 longlong3_h[16];
  // CHECK: cl::sycl::longlong3 longlong3_i[32];
  longlong3 longlong3_i[32];
  // CHECK: if (static_cast<long long>(longlong3_h[12].x()) == static_cast<long long>(longlong3_i[12].x())) {}
  if (longlong3_h[12].x == longlong3_i[12].x) {}
  // CHECK: longlong3_f = (cl::sycl::longlong3 *)longlong3_i;
  longlong3_f = (longlong3 *)longlong3_i;
  // CHECK: longlong3_a = (cl::sycl::longlong3)longlong3_c;
  longlong3_a = (longlong3)longlong3_c;
  // CHECK: longlong3_b = cl::sycl::longlong3(longlong3_b);
  longlong3_b = longlong3(longlong3_b);
  // CHECK: cl::sycl::longlong3 longlong3_j, longlong3_k, longlong3_l, longlong3_m[16], *longlong3_n[32];
  longlong3 longlong3_j, longlong3_k, longlong3_l, longlong3_m[16], *longlong3_n[32];
  // CHECK: int longlong3_o = sizeof(cl::sycl::longlong3);
  int longlong3_o = sizeof(longlong3);
  // CHECK: int long long_p = sizeof(long long);
  int long long_p = sizeof(long long);
  // CHECK: int longlong3_q = sizeof(longlong3_d);
  int longlong3_q = sizeof(longlong3_d);
  int *longlong3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> longlong3_e_buf_ct0 = dpct::get_buffer_and_offset(longlong3_e);
  // CHECK-NEXT:   size_t longlong3_e_offset_ct0 = longlong3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> longlong3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::longlong3 *)longlong3_cast);
  // CHECK-NEXT:   size_t longlong3_cast_offset_ct1 = longlong3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto longlong3_e_acc_ct0 = longlong3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto longlong3_cast_acc_ct1 = longlong3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_longlong3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::longlong3 *longlong3_e_ct0 = (cl::sycl::longlong3 *)(&longlong3_e_acc_ct0[0] + longlong3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::longlong3 *longlong3_cast_ct1 = (cl::sycl::longlong3 *)(&longlong3_cast_acc_ct1[0] + longlong3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_longlong3(longlong3_e_ct0, longlong3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_longlong3<<<1,1>>>(longlong3_e, (longlong3 *)longlong3_cast);
  return 0;
}

// CHECK: void func3_longlong4(cl::sycl::longlong4 a, cl::sycl::longlong4 b, cl::sycl::longlong4 c) {
void func3_longlong4(longlong4 a, longlong4 b, longlong4 c) {
}
// CHECK: void func_longlong4(cl::sycl::longlong4 a) {
void func_longlong4(longlong4 a) {
}
// CHECK: void kernel_longlong4(cl::sycl::longlong4 *a, cl::sycl::longlong4 *b) {
__global__ void kernel_longlong4(longlong4 *a, longlong4 *b) {
}

int main_longlong4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::longlong4 longlong4_a;
  longlong4 longlong4_a;
  // CHECK: cl::sycl::longlong4 longlong4_b = cl::sycl::longlong4(1, 2, 3, 4);
  longlong4 longlong4_b = make_longlong4(1, 2, 3, 4);
  // CHECK: cl::sycl::longlong4 longlong4_c = cl::sycl::longlong4(longlong4_b);
  longlong4 longlong4_c = longlong4(longlong4_b);
  // CHECK: cl::sycl::longlong4 longlong4_d(longlong4_c);
  longlong4 longlong4_d(longlong4_c);
  // CHECK: func3_longlong4(longlong4_b, cl::sycl::longlong4(longlong4_b), (cl::sycl::longlong4)longlong4_b);
  func3_longlong4(longlong4_b, longlong4(longlong4_b), (longlong4)longlong4_b);
  // CHECK: cl::sycl::longlong4 *longlong4_e;
  longlong4 *longlong4_e;
  // CHECK: cl::sycl::longlong4 *longlong4_f;
  longlong4 *longlong4_f;
  // CHECK: long long longlong4_g = static_cast<long long>(longlong4_c.x());
  long long longlong4_g = longlong4_c.x;
  // CHECK: longlong4_a.x() = static_cast<long long>(longlong4_d.x());
  longlong4_a.x = longlong4_d.x;
  // CHECK: if (static_cast<long long>(longlong4_b.x()) == static_cast<long long>(longlong4_d.x())) {}
  if (longlong4_b.x == longlong4_d.x) {}
  // CHECK: cl::sycl::longlong4 longlong4_h[16];
  longlong4 longlong4_h[16];
  // CHECK: cl::sycl::longlong4 longlong4_i[32];
  longlong4 longlong4_i[32];
  // CHECK: if (static_cast<long long>(longlong4_h[12].x()) == static_cast<long long>(longlong4_i[12].x())) {}
  if (longlong4_h[12].x == longlong4_i[12].x) {}
  // CHECK: longlong4_f = (cl::sycl::longlong4 *)longlong4_i;
  longlong4_f = (longlong4 *)longlong4_i;
  // CHECK: longlong4_a = (cl::sycl::longlong4)longlong4_c;
  longlong4_a = (longlong4)longlong4_c;
  // CHECK: longlong4_b = cl::sycl::longlong4(longlong4_b);
  longlong4_b = longlong4(longlong4_b);
  // CHECK: cl::sycl::longlong4 longlong4_j, longlong4_k, longlong4_l, longlong4_m[16], *longlong4_n[32];
  longlong4 longlong4_j, longlong4_k, longlong4_l, longlong4_m[16], *longlong4_n[32];
  // CHECK: int longlong4_o = sizeof(cl::sycl::longlong4);
  int longlong4_o = sizeof(longlong4);
  // CHECK: int long long_p = sizeof(long long);
  int long long_p = sizeof(long long);
  // CHECK: int longlong4_q = sizeof(longlong4_d);
  int longlong4_q = sizeof(longlong4_d);
  int *longlong4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> longlong4_e_buf_ct0 = dpct::get_buffer_and_offset(longlong4_e);
  // CHECK-NEXT:   size_t longlong4_e_offset_ct0 = longlong4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> longlong4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::longlong4 *)longlong4_cast);
  // CHECK-NEXT:   size_t longlong4_cast_offset_ct1 = longlong4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto longlong4_e_acc_ct0 = longlong4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto longlong4_cast_acc_ct1 = longlong4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_longlong4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::longlong4 *longlong4_e_ct0 = (cl::sycl::longlong4 *)(&longlong4_e_acc_ct0[0] + longlong4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::longlong4 *longlong4_cast_ct1 = (cl::sycl::longlong4 *)(&longlong4_cast_acc_ct1[0] + longlong4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_longlong4(longlong4_e_ct0, longlong4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_longlong4<<<1,1>>>(longlong4_e, (longlong4 *)longlong4_cast);
  return 0;
}

// CHECK: void func3_short1(short a, short b, short c) {
void func3_short1(short1 a, short1 b, short1 c) {
}
// CHECK: void func_short1(short a) {
void func_short1(short1 a) {
}
// CHECK: void kernel_short1(short *a, short *b) {
__global__ void kernel_short1(short1 *a, short1 *b) {
}

int main_short1() {
  // range default constructor does the right thing.
  // CHECK: short short1_a;
  short1 short1_a;
  // CHECK: short short1_b = short(1);
  short1 short1_b = make_short1(1);
  // CHECK: short short1_c = short(short1_b);
  short1 short1_c = short1(short1_b);
  // CHECK: short short1_d(short1_c);
  short1 short1_d(short1_c);
  // CHECK: func3_short1(short1_b, short(short1_b), (short)short1_b);
  func3_short1(short1_b, short1(short1_b), (short1)short1_b);
  // CHECK: short *short1_e;
  short1 *short1_e;
  // CHECK: short *short1_f;
  short1 *short1_f;
  // CHECK: short short1_g = static_cast<short>(short1_c);
  short short1_g = short1_c.x;
  // CHECK: short1_a = static_cast<short>(short1_d);
  short1_a.x = short1_d.x;
  // CHECK: if (static_cast<short>(short1_b) == static_cast<short>(short1_d)) {}
  if (short1_b.x == short1_d.x) {}
  // CHECK: short short1_h[16];
  short1 short1_h[16];
  // CHECK: short short1_i[32];
  short1 short1_i[32];
  // CHECK: if (static_cast<short>(short1_h[12]) == static_cast<short>(short1_i[12])) {}
  if (short1_h[12].x == short1_i[12].x) {}
  // CHECK: short1_f = (short *)short1_i;
  short1_f = (short1 *)short1_i;
  // CHECK: short1_a = (short)short1_c;
  short1_a = (short1)short1_c;
  // CHECK: short1_b = short(short1_b);
  short1_b = short1(short1_b);
  // CHECK: short short1_j, short1_k, short1_l, short1_m[16], *short1_n[32];
  short1 short1_j, short1_k, short1_l, short1_m[16], *short1_n[32];
  // CHECK: int short1_o = sizeof(short);
  int short1_o = sizeof(short1);
  // CHECK: int short_p = sizeof(short);
  int short_p = sizeof(short);
  // CHECK: int short1_q = sizeof(short1_d);
  int short1_q = sizeof(short1_d);
  int *short1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> short1_e_buf_ct0 = dpct::get_buffer_and_offset(short1_e);
  // CHECK-NEXT:   size_t short1_e_offset_ct0 = short1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> short1_cast_buf_ct1 = dpct::get_buffer_and_offset((short *)short1_cast);
  // CHECK-NEXT:   size_t short1_cast_offset_ct1 = short1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto short1_e_acc_ct0 = short1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto short1_cast_acc_ct1 = short1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_short1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           short *short1_e_ct0 = (short *)(&short1_e_acc_ct0[0] + short1_e_offset_ct0);
  // CHECK-NEXT:           short *short1_cast_ct1 = (short *)(&short1_cast_acc_ct1[0] + short1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_short1(short1_e_ct0, short1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_short1<<<1,1>>>(short1_e, (short1 *)short1_cast);
  return 0;
}

// CHECK: void func3_short2(cl::sycl::short2 a, cl::sycl::short2 b, cl::sycl::short2 c) {
void func3_short2(short2 a, short2 b, short2 c) {
}
// CHECK: void func_short2(cl::sycl::short2 a) {
void func_short2(short2 a) {
}
// CHECK: void kernel_short2(cl::sycl::short2 *a, cl::sycl::short2 *b) {
__global__ void kernel_short2(short2 *a, short2 *b) {
}

int main_short2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::short2 short2_a;
  short2 short2_a;
  // CHECK: cl::sycl::short2 short2_b = cl::sycl::short2(1, 2);
  short2 short2_b = make_short2(1, 2);
  // CHECK: cl::sycl::short2 short2_c = cl::sycl::short2(short2_b);
  short2 short2_c = short2(short2_b);
  // CHECK: cl::sycl::short2 short2_d(short2_c);
  short2 short2_d(short2_c);
  // CHECK: func3_short2(short2_b, cl::sycl::short2(short2_b), (cl::sycl::short2)short2_b);
  func3_short2(short2_b, short2(short2_b), (short2)short2_b);
  // CHECK: cl::sycl::short2 *short2_e;
  short2 *short2_e;
  // CHECK: cl::sycl::short2 *short2_f;
  short2 *short2_f;
  // CHECK: short short2_g = static_cast<short>(short2_c.x());
  short short2_g = short2_c.x;
  // CHECK: short2_a.x() = static_cast<short>(short2_d.x());
  short2_a.x = short2_d.x;
  // CHECK: if (static_cast<short>(short2_b.x()) == static_cast<short>(short2_d.x())) {}
  if (short2_b.x == short2_d.x) {}
  // CHECK: cl::sycl::short2 short2_h[16];
  short2 short2_h[16];
  // CHECK: cl::sycl::short2 short2_i[32];
  short2 short2_i[32];
  // CHECK: if (static_cast<short>(short2_h[12].x()) == static_cast<short>(short2_i[12].x())) {}
  if (short2_h[12].x == short2_i[12].x) {}
  // CHECK: short2_f = (cl::sycl::short2 *)short2_i;
  short2_f = (short2 *)short2_i;
  // CHECK: short2_a = (cl::sycl::short2)short2_c;
  short2_a = (short2)short2_c;
  // CHECK: short2_b = cl::sycl::short2(short2_b);
  short2_b = short2(short2_b);
  // CHECK: cl::sycl::short2 short2_j, short2_k, short2_l, short2_m[16], *short2_n[32];
  short2 short2_j, short2_k, short2_l, short2_m[16], *short2_n[32];
  // CHECK: int short2_o = sizeof(cl::sycl::short2);
  int short2_o = sizeof(short2);
  // CHECK: int short_p = sizeof(short);
  int short_p = sizeof(short);
  // CHECK: int short2_q = sizeof(short2_d);
  int short2_q = sizeof(short2_d);
  int *short2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> short2_e_buf_ct0 = dpct::get_buffer_and_offset(short2_e);
  // CHECK-NEXT:   size_t short2_e_offset_ct0 = short2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> short2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::short2 *)short2_cast);
  // CHECK-NEXT:   size_t short2_cast_offset_ct1 = short2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto short2_e_acc_ct0 = short2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto short2_cast_acc_ct1 = short2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_short2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::short2 *short2_e_ct0 = (cl::sycl::short2 *)(&short2_e_acc_ct0[0] + short2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::short2 *short2_cast_ct1 = (cl::sycl::short2 *)(&short2_cast_acc_ct1[0] + short2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_short2(short2_e_ct0, short2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_short2<<<1,1>>>(short2_e, (short2 *)short2_cast);
  return 0;
}

// CHECK: void func3_short3(cl::sycl::short3 a, cl::sycl::short3 b, cl::sycl::short3 c) {
void func3_short3(short3 a, short3 b, short3 c) {
}
// CHECK: void func_short3(cl::sycl::short3 a) {
void func_short3(short3 a) {
}
// CHECK: void kernel_short3(cl::sycl::short3 *a, cl::sycl::short3 *b) {
__global__ void kernel_short3(short3 *a, short3 *b) {
}

int main_short3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::short3 short3_a;
  short3 short3_a;
  // CHECK: cl::sycl::short3 short3_b = cl::sycl::short3(1, 2, 3);
  short3 short3_b = make_short3(1, 2, 3);
  // CHECK: cl::sycl::short3 short3_c = cl::sycl::short3(short3_b);
  short3 short3_c = short3(short3_b);
  // CHECK: cl::sycl::short3 short3_d(short3_c);
  short3 short3_d(short3_c);
  // CHECK: func3_short3(short3_b, cl::sycl::short3(short3_b), (cl::sycl::short3)short3_b);
  func3_short3(short3_b, short3(short3_b), (short3)short3_b);
  // CHECK: cl::sycl::short3 *short3_e;
  short3 *short3_e;
  // CHECK: cl::sycl::short3 *short3_f;
  short3 *short3_f;
  // CHECK: short short3_g = static_cast<short>(short3_c.x());
  short short3_g = short3_c.x;
  // CHECK: short3_a.x() = static_cast<short>(short3_d.x());
  short3_a.x = short3_d.x;
  // CHECK: if (static_cast<short>(short3_b.x()) == static_cast<short>(short3_d.x())) {}
  if (short3_b.x == short3_d.x) {}
  // CHECK: cl::sycl::short3 short3_h[16];
  short3 short3_h[16];
  // CHECK: cl::sycl::short3 short3_i[32];
  short3 short3_i[32];
  // CHECK: if (static_cast<short>(short3_h[12].x()) == static_cast<short>(short3_i[12].x())) {}
  if (short3_h[12].x == short3_i[12].x) {}
  // CHECK: short3_f = (cl::sycl::short3 *)short3_i;
  short3_f = (short3 *)short3_i;
  // CHECK: short3_a = (cl::sycl::short3)short3_c;
  short3_a = (short3)short3_c;
  // CHECK: short3_b = cl::sycl::short3(short3_b);
  short3_b = short3(short3_b);
  // CHECK: cl::sycl::short3 short3_j, short3_k, short3_l, short3_m[16], *short3_n[32];
  short3 short3_j, short3_k, short3_l, short3_m[16], *short3_n[32];
  // CHECK: int short3_o = sizeof(cl::sycl::short3);
  int short3_o = sizeof(short3);
  // CHECK: int short_p = sizeof(short);
  int short_p = sizeof(short);
  // CHECK: int short3_q = sizeof(short3_d);
  int short3_q = sizeof(short3_d);
  int *short3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> short3_e_buf_ct0 = dpct::get_buffer_and_offset(short3_e);
  // CHECK-NEXT:   size_t short3_e_offset_ct0 = short3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> short3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::short3 *)short3_cast);
  // CHECK-NEXT:   size_t short3_cast_offset_ct1 = short3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto short3_e_acc_ct0 = short3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto short3_cast_acc_ct1 = short3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_short3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::short3 *short3_e_ct0 = (cl::sycl::short3 *)(&short3_e_acc_ct0[0] + short3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::short3 *short3_cast_ct1 = (cl::sycl::short3 *)(&short3_cast_acc_ct1[0] + short3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_short3(short3_e_ct0, short3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_short3<<<1,1>>>(short3_e, (short3 *)short3_cast);
  return 0;
}

// CHECK: void func3_short4(cl::sycl::short4 a, cl::sycl::short4 b, cl::sycl::short4 c) {
void func3_short4(short4 a, short4 b, short4 c) {
}
// CHECK: void func_short4(cl::sycl::short4 a) {
void func_short4(short4 a) {
}
// CHECK: void kernel_short4(cl::sycl::short4 *a, cl::sycl::short4 *b) {
__global__ void kernel_short4(short4 *a, short4 *b) {
}

int main_short4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::short4 short4_a;
  short4 short4_a;
  // CHECK: cl::sycl::short4 short4_b = cl::sycl::short4(1, 2, 3, 4);
  short4 short4_b = make_short4(1, 2, 3, 4);
  // CHECK: cl::sycl::short4 short4_c = cl::sycl::short4(short4_b);
  short4 short4_c = short4(short4_b);
  // CHECK: cl::sycl::short4 short4_d(short4_c);
  short4 short4_d(short4_c);
  // CHECK: func3_short4(short4_b, cl::sycl::short4(short4_b), (cl::sycl::short4)short4_b);
  func3_short4(short4_b, short4(short4_b), (short4)short4_b);
  // CHECK: cl::sycl::short4 *short4_e;
  short4 *short4_e;
  // CHECK: cl::sycl::short4 *short4_f;
  short4 *short4_f;
  // CHECK: short short4_g = static_cast<short>(short4_c.x());
  short short4_g = short4_c.x;
  // CHECK: short4_a.x() = static_cast<short>(short4_d.x());
  short4_a.x = short4_d.x;
  // CHECK: if (static_cast<short>(short4_b.x()) == static_cast<short>(short4_d.x())) {}
  if (short4_b.x == short4_d.x) {}
  // CHECK: cl::sycl::short4 short4_h[16];
  short4 short4_h[16];
  // CHECK: cl::sycl::short4 short4_i[32];
  short4 short4_i[32];
  // CHECK: if (static_cast<short>(short4_h[12].x()) == static_cast<short>(short4_i[12].x())) {}
  if (short4_h[12].x == short4_i[12].x) {}
  // CHECK: short4_f = (cl::sycl::short4 *)short4_i;
  short4_f = (short4 *)short4_i;
  // CHECK: short4_a = (cl::sycl::short4)short4_c;
  short4_a = (short4)short4_c;
  // CHECK: short4_b = cl::sycl::short4(short4_b);
  short4_b = short4(short4_b);
  // CHECK: cl::sycl::short4 short4_j, short4_k, short4_l, short4_m[16], *short4_n[32];
  short4 short4_j, short4_k, short4_l, short4_m[16], *short4_n[32];
  // CHECK: int short4_o = sizeof(cl::sycl::short4);
  int short4_o = sizeof(short4);
  // CHECK: int short_p = sizeof(short);
  int short_p = sizeof(short);
  // CHECK: int short4_q = sizeof(short4_d);
  int short4_q = sizeof(short4_d);
  int *short4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> short4_e_buf_ct0 = dpct::get_buffer_and_offset(short4_e);
  // CHECK-NEXT:   size_t short4_e_offset_ct0 = short4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> short4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::short4 *)short4_cast);
  // CHECK-NEXT:   size_t short4_cast_offset_ct1 = short4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto short4_e_acc_ct0 = short4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto short4_cast_acc_ct1 = short4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_short4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::short4 *short4_e_ct0 = (cl::sycl::short4 *)(&short4_e_acc_ct0[0] + short4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::short4 *short4_cast_ct1 = (cl::sycl::short4 *)(&short4_cast_acc_ct1[0] + short4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_short4(short4_e_ct0, short4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_short4<<<1,1>>>(short4_e, (short4 *)short4_cast);
  return 0;
}

// CHECK: void func3_uchar1(unsigned char a, unsigned char b, unsigned char c) {
void func3_uchar1(uchar1 a, uchar1 b, uchar1 c) {
}
// CHECK: void func_uchar1(unsigned char a) {
void func_uchar1(uchar1 a) {
}
// CHECK: void kernel_uchar1(unsigned char *a, unsigned char *b) {
__global__ void kernel_uchar1(uchar1 *a, uchar1 *b) {
}

int main_uchar1() {
  // range default constructor does the right thing.
  // CHECK: unsigned char uchar1_a;
  uchar1 uchar1_a;
  // CHECK: unsigned char uchar1_b = unsigned char(1);
  uchar1 uchar1_b = make_uchar1(1);
  // CHECK: unsigned char uchar1_c = unsigned char(uchar1_b);
  uchar1 uchar1_c = uchar1(uchar1_b);
  // CHECK: unsigned char uchar1_d(uchar1_c);
  uchar1 uchar1_d(uchar1_c);
  // CHECK: func3_uchar1(uchar1_b, unsigned char(uchar1_b), (unsigned char)uchar1_b);
  func3_uchar1(uchar1_b, uchar1(uchar1_b), (uchar1)uchar1_b);
  // CHECK: unsigned char *uchar1_e;
  uchar1 *uchar1_e;
  // CHECK: unsigned char *uchar1_f;
  uchar1 *uchar1_f;
  // CHECK: unsigned char uchar1_g = static_cast<unsigned char>(uchar1_c);
  unsigned char uchar1_g = uchar1_c.x;
  // CHECK: uchar1_a = static_cast<unsigned char>(uchar1_d);
  uchar1_a.x = uchar1_d.x;
  // CHECK: if (static_cast<unsigned char>(uchar1_b) == static_cast<unsigned char>(uchar1_d)) {}
  if (uchar1_b.x == uchar1_d.x) {}
  // CHECK: unsigned char uchar1_h[16];
  uchar1 uchar1_h[16];
  // CHECK: unsigned char uchar1_i[32];
  uchar1 uchar1_i[32];
  // CHECK: if (static_cast<unsigned char>(uchar1_h[12]) == static_cast<unsigned char>(uchar1_i[12])) {}
  if (uchar1_h[12].x == uchar1_i[12].x) {}
  // CHECK: uchar1_f = (unsigned char *)uchar1_i;
  uchar1_f = (uchar1 *)uchar1_i;
  // CHECK: uchar1_a = (unsigned char)uchar1_c;
  uchar1_a = (uchar1)uchar1_c;
  // CHECK: uchar1_b = unsigned char(uchar1_b);
  uchar1_b = uchar1(uchar1_b);
  // CHECK: unsigned char uchar1_j, uchar1_k, uchar1_l, uchar1_m[16], *uchar1_n[32];
  uchar1 uchar1_j, uchar1_k, uchar1_l, uchar1_m[16], *uchar1_n[32];
  // CHECK: int uchar1_o = sizeof(unsigned char);
  int uchar1_o = sizeof(uchar1);
  // CHECK: int unsigned char_p = sizeof(unsigned char);
  int unsigned char_p = sizeof(unsigned char);
  // CHECK: int uchar1_q = sizeof(uchar1_d);
  int uchar1_q = sizeof(uchar1_d);
  int *uchar1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uchar1_e_buf_ct0 = dpct::get_buffer_and_offset(uchar1_e);
  // CHECK-NEXT:   size_t uchar1_e_offset_ct0 = uchar1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uchar1_cast_buf_ct1 = dpct::get_buffer_and_offset((unsigned char *)uchar1_cast);
  // CHECK-NEXT:   size_t uchar1_cast_offset_ct1 = uchar1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto uchar1_e_acc_ct0 = uchar1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto uchar1_cast_acc_ct1 = uchar1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_uchar1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           unsigned char *uchar1_e_ct0 = (unsigned char *)(&uchar1_e_acc_ct0[0] + uchar1_e_offset_ct0);
  // CHECK-NEXT:           unsigned char *uchar1_cast_ct1 = (unsigned char *)(&uchar1_cast_acc_ct1[0] + uchar1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_uchar1(uchar1_e_ct0, uchar1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_uchar1<<<1,1>>>(uchar1_e, (uchar1 *)uchar1_cast);
  return 0;
}

// CHECK: void func3_uchar2(cl::sycl::uchar2 a, cl::sycl::uchar2 b, cl::sycl::uchar2 c) {
void func3_uchar2(uchar2 a, uchar2 b, uchar2 c) {
}
// CHECK: void func_uchar2(cl::sycl::uchar2 a) {
void func_uchar2(uchar2 a) {
}
// CHECK: void kernel_uchar2(cl::sycl::uchar2 *a, cl::sycl::uchar2 *b) {
__global__ void kernel_uchar2(uchar2 *a, uchar2 *b) {
}

int main_uchar2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uchar2 uchar2_a;
  uchar2 uchar2_a;
  // CHECK: cl::sycl::uchar2 uchar2_b = cl::sycl::uchar2(1, 2);
  uchar2 uchar2_b = make_uchar2(1, 2);
  // CHECK: cl::sycl::uchar2 uchar2_c = cl::sycl::uchar2(uchar2_b);
  uchar2 uchar2_c = uchar2(uchar2_b);
  // CHECK: cl::sycl::uchar2 uchar2_d(uchar2_c);
  uchar2 uchar2_d(uchar2_c);
  // CHECK: func3_uchar2(uchar2_b, cl::sycl::uchar2(uchar2_b), (cl::sycl::uchar2)uchar2_b);
  func3_uchar2(uchar2_b, uchar2(uchar2_b), (uchar2)uchar2_b);
  // CHECK: cl::sycl::uchar2 *uchar2_e;
  uchar2 *uchar2_e;
  // CHECK: cl::sycl::uchar2 *uchar2_f;
  uchar2 *uchar2_f;
  // CHECK: unsigned char uchar2_g = static_cast<unsigned char>(uchar2_c.x());
  unsigned char uchar2_g = uchar2_c.x;
  // CHECK: uchar2_a.x() = static_cast<unsigned char>(uchar2_d.x());
  uchar2_a.x = uchar2_d.x;
  // CHECK: if (static_cast<unsigned char>(uchar2_b.x()) == static_cast<unsigned char>(uchar2_d.x())) {}
  if (uchar2_b.x == uchar2_d.x) {}
  // CHECK: cl::sycl::uchar2 uchar2_h[16];
  uchar2 uchar2_h[16];
  // CHECK: cl::sycl::uchar2 uchar2_i[32];
  uchar2 uchar2_i[32];
  // CHECK: if (static_cast<unsigned char>(uchar2_h[12].x()) == static_cast<unsigned char>(uchar2_i[12].x())) {}
  if (uchar2_h[12].x == uchar2_i[12].x) {}
  // CHECK: uchar2_f = (cl::sycl::uchar2 *)uchar2_i;
  uchar2_f = (uchar2 *)uchar2_i;
  // CHECK: uchar2_a = (cl::sycl::uchar2)uchar2_c;
  uchar2_a = (uchar2)uchar2_c;
  // CHECK: uchar2_b = cl::sycl::uchar2(uchar2_b);
  uchar2_b = uchar2(uchar2_b);
  // CHECK: cl::sycl::uchar2 uchar2_j, uchar2_k, uchar2_l, uchar2_m[16], *uchar2_n[32];
  uchar2 uchar2_j, uchar2_k, uchar2_l, uchar2_m[16], *uchar2_n[32];
  // CHECK: int uchar2_o = sizeof(cl::sycl::uchar2);
  int uchar2_o = sizeof(uchar2);
  // CHECK: int unsigned char_p = sizeof(unsigned char);
  int unsigned char_p = sizeof(unsigned char);
  // CHECK: int uchar2_q = sizeof(uchar2_d);
  int uchar2_q = sizeof(uchar2_d);
  int *uchar2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uchar2_e_buf_ct0 = dpct::get_buffer_and_offset(uchar2_e);
  // CHECK-NEXT:   size_t uchar2_e_offset_ct0 = uchar2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uchar2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::uchar2 *)uchar2_cast);
  // CHECK-NEXT:   size_t uchar2_cast_offset_ct1 = uchar2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto uchar2_e_acc_ct0 = uchar2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto uchar2_cast_acc_ct1 = uchar2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_uchar2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::uchar2 *uchar2_e_ct0 = (cl::sycl::uchar2 *)(&uchar2_e_acc_ct0[0] + uchar2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::uchar2 *uchar2_cast_ct1 = (cl::sycl::uchar2 *)(&uchar2_cast_acc_ct1[0] + uchar2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_uchar2(uchar2_e_ct0, uchar2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_uchar2<<<1,1>>>(uchar2_e, (uchar2 *)uchar2_cast);
  return 0;
}

// CHECK: void func3_uchar3(cl::sycl::uchar3 a, cl::sycl::uchar3 b, cl::sycl::uchar3 c) {
void func3_uchar3(uchar3 a, uchar3 b, uchar3 c) {
}
// CHECK: void func_uchar3(cl::sycl::uchar3 a) {
void func_uchar3(uchar3 a) {
}
// CHECK: void kernel_uchar3(cl::sycl::uchar3 *a, cl::sycl::uchar3 *b) {
__global__ void kernel_uchar3(uchar3 *a, uchar3 *b) {
}

int main_uchar3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uchar3 uchar3_a;
  uchar3 uchar3_a;
  // CHECK: cl::sycl::uchar3 uchar3_b = cl::sycl::uchar3(1, 2, 3);
  uchar3 uchar3_b = make_uchar3(1, 2, 3);
  // CHECK: cl::sycl::uchar3 uchar3_c = cl::sycl::uchar3(uchar3_b);
  uchar3 uchar3_c = uchar3(uchar3_b);
  // CHECK: cl::sycl::uchar3 uchar3_d(uchar3_c);
  uchar3 uchar3_d(uchar3_c);
  // CHECK: func3_uchar3(uchar3_b, cl::sycl::uchar3(uchar3_b), (cl::sycl::uchar3)uchar3_b);
  func3_uchar3(uchar3_b, uchar3(uchar3_b), (uchar3)uchar3_b);
  // CHECK: cl::sycl::uchar3 *uchar3_e;
  uchar3 *uchar3_e;
  // CHECK: cl::sycl::uchar3 *uchar3_f;
  uchar3 *uchar3_f;
  // CHECK: unsigned char uchar3_g = static_cast<unsigned char>(uchar3_c.x());
  unsigned char uchar3_g = uchar3_c.x;
  // CHECK: uchar3_a.x() = static_cast<unsigned char>(uchar3_d.x());
  uchar3_a.x = uchar3_d.x;
  // CHECK: if (static_cast<unsigned char>(uchar3_b.x()) == static_cast<unsigned char>(uchar3_d.x())) {}
  if (uchar3_b.x == uchar3_d.x) {}
  // CHECK: cl::sycl::uchar3 uchar3_h[16];
  uchar3 uchar3_h[16];
  // CHECK: cl::sycl::uchar3 uchar3_i[32];
  uchar3 uchar3_i[32];
  // CHECK: if (static_cast<unsigned char>(uchar3_h[12].x()) == static_cast<unsigned char>(uchar3_i[12].x())) {}
  if (uchar3_h[12].x == uchar3_i[12].x) {}
  // CHECK: uchar3_f = (cl::sycl::uchar3 *)uchar3_i;
  uchar3_f = (uchar3 *)uchar3_i;
  // CHECK: uchar3_a = (cl::sycl::uchar3)uchar3_c;
  uchar3_a = (uchar3)uchar3_c;
  // CHECK: uchar3_b = cl::sycl::uchar3(uchar3_b);
  uchar3_b = uchar3(uchar3_b);
  // CHECK: cl::sycl::uchar3 uchar3_j, uchar3_k, uchar3_l, uchar3_m[16], *uchar3_n[32];
  uchar3 uchar3_j, uchar3_k, uchar3_l, uchar3_m[16], *uchar3_n[32];
  // CHECK: int uchar3_o = sizeof(cl::sycl::uchar3);
  int uchar3_o = sizeof(uchar3);
  // CHECK: int unsigned char_p = sizeof(unsigned char);
  int unsigned char_p = sizeof(unsigned char);
  // CHECK: int uchar3_q = sizeof(uchar3_d);
  int uchar3_q = sizeof(uchar3_d);
  int *uchar3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uchar3_e_buf_ct0 = dpct::get_buffer_and_offset(uchar3_e);
  // CHECK-NEXT:   size_t uchar3_e_offset_ct0 = uchar3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uchar3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::uchar3 *)uchar3_cast);
  // CHECK-NEXT:   size_t uchar3_cast_offset_ct1 = uchar3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto uchar3_e_acc_ct0 = uchar3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto uchar3_cast_acc_ct1 = uchar3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_uchar3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::uchar3 *uchar3_e_ct0 = (cl::sycl::uchar3 *)(&uchar3_e_acc_ct0[0] + uchar3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::uchar3 *uchar3_cast_ct1 = (cl::sycl::uchar3 *)(&uchar3_cast_acc_ct1[0] + uchar3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_uchar3(uchar3_e_ct0, uchar3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_uchar3<<<1,1>>>(uchar3_e, (uchar3 *)uchar3_cast);
  return 0;
}

// CHECK: void func3_uchar4(cl::sycl::uchar4 a, cl::sycl::uchar4 b, cl::sycl::uchar4 c) {
void func3_uchar4(uchar4 a, uchar4 b, uchar4 c) {
}
// CHECK: void func_uchar4(cl::sycl::uchar4 a) {
void func_uchar4(uchar4 a) {
}
// CHECK: void kernel_uchar4(cl::sycl::uchar4 *a, cl::sycl::uchar4 *b) {
__global__ void kernel_uchar4(uchar4 *a, uchar4 *b) {
}

int main_uchar4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uchar4 uchar4_a;
  uchar4 uchar4_a;
  // CHECK: cl::sycl::uchar4 uchar4_b = cl::sycl::uchar4(1, 2, 3, 4);
  uchar4 uchar4_b = make_uchar4(1, 2, 3, 4);
  // CHECK: cl::sycl::uchar4 uchar4_c = cl::sycl::uchar4(uchar4_b);
  uchar4 uchar4_c = uchar4(uchar4_b);
  // CHECK: cl::sycl::uchar4 uchar4_d(uchar4_c);
  uchar4 uchar4_d(uchar4_c);
  // CHECK: func3_uchar4(uchar4_b, cl::sycl::uchar4(uchar4_b), (cl::sycl::uchar4)uchar4_b);
  func3_uchar4(uchar4_b, uchar4(uchar4_b), (uchar4)uchar4_b);
  // CHECK: cl::sycl::uchar4 *uchar4_e;
  uchar4 *uchar4_e;
  // CHECK: cl::sycl::uchar4 *uchar4_f;
  uchar4 *uchar4_f;
  // CHECK: unsigned char uchar4_g = static_cast<unsigned char>(uchar4_c.x());
  unsigned char uchar4_g = uchar4_c.x;
  // CHECK: uchar4_a.x() = static_cast<unsigned char>(uchar4_d.x());
  uchar4_a.x = uchar4_d.x;
  // CHECK: if (static_cast<unsigned char>(uchar4_b.x()) == static_cast<unsigned char>(uchar4_d.x())) {}
  if (uchar4_b.x == uchar4_d.x) {}
  // CHECK: cl::sycl::uchar4 uchar4_h[16];
  uchar4 uchar4_h[16];
  // CHECK: cl::sycl::uchar4 uchar4_i[32];
  uchar4 uchar4_i[32];
  // CHECK: if (static_cast<unsigned char>(uchar4_h[12].x()) == static_cast<unsigned char>(uchar4_i[12].x())) {}
  if (uchar4_h[12].x == uchar4_i[12].x) {}
  // CHECK: uchar4_f = (cl::sycl::uchar4 *)uchar4_i;
  uchar4_f = (uchar4 *)uchar4_i;
  // CHECK: uchar4_a = (cl::sycl::uchar4)uchar4_c;
  uchar4_a = (uchar4)uchar4_c;
  // CHECK: uchar4_b = cl::sycl::uchar4(uchar4_b);
  uchar4_b = uchar4(uchar4_b);
  // CHECK: cl::sycl::uchar4 uchar4_j, uchar4_k, uchar4_l, uchar4_m[16], *uchar4_n[32];
  uchar4 uchar4_j, uchar4_k, uchar4_l, uchar4_m[16], *uchar4_n[32];
  // CHECK: int uchar4_o = sizeof(cl::sycl::uchar4);
  int uchar4_o = sizeof(uchar4);
  // CHECK: int unsigned char_p = sizeof(unsigned char);
  int unsigned char_p = sizeof(unsigned char);
  // CHECK: int uchar4_q = sizeof(uchar4_d);
  int uchar4_q = sizeof(uchar4_d);
  int *uchar4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uchar4_e_buf_ct0 = dpct::get_buffer_and_offset(uchar4_e);
  // CHECK-NEXT:   size_t uchar4_e_offset_ct0 = uchar4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uchar4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::uchar4 *)uchar4_cast);
  // CHECK-NEXT:   size_t uchar4_cast_offset_ct1 = uchar4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto uchar4_e_acc_ct0 = uchar4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto uchar4_cast_acc_ct1 = uchar4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_uchar4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::uchar4 *uchar4_e_ct0 = (cl::sycl::uchar4 *)(&uchar4_e_acc_ct0[0] + uchar4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::uchar4 *uchar4_cast_ct1 = (cl::sycl::uchar4 *)(&uchar4_cast_acc_ct1[0] + uchar4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_uchar4(uchar4_e_ct0, uchar4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_uchar4<<<1,1>>>(uchar4_e, (uchar4 *)uchar4_cast);
  return 0;
}

// CHECK: void func3_uint1(unsigned int a, unsigned int b, unsigned int c) {
void func3_uint1(uint1 a, uint1 b, uint1 c) {
}
// CHECK: void func_uint1(unsigned int a) {
void func_uint1(uint1 a) {
}
// CHECK: void kernel_uint1(unsigned int *a, unsigned int *b) {
__global__ void kernel_uint1(uint1 *a, uint1 *b) {
}

int main_uint1() {
  // range default constructor does the right thing.
  // CHECK: unsigned int uint1_a;
  uint1 uint1_a;
  // CHECK: unsigned int uint1_b = unsigned int(1);
  uint1 uint1_b = make_uint1(1);
  // CHECK: unsigned int uint1_c = unsigned int(uint1_b);
  uint1 uint1_c = uint1(uint1_b);
  // CHECK: unsigned int uint1_d(uint1_c);
  uint1 uint1_d(uint1_c);
  // CHECK: func3_uint1(uint1_b, unsigned int(uint1_b), (unsigned int)uint1_b);
  func3_uint1(uint1_b, uint1(uint1_b), (uint1)uint1_b);
  // CHECK: unsigned int *uint1_e;
  uint1 *uint1_e;
  // CHECK: unsigned int *uint1_f;
  uint1 *uint1_f;
  // CHECK: unsigned int uint1_g = static_cast<unsigned int>(uint1_c);
  unsigned int uint1_g = uint1_c.x;
  // CHECK: uint1_a = static_cast<unsigned int>(uint1_d);
  uint1_a.x = uint1_d.x;
  // CHECK: if (static_cast<unsigned int>(uint1_b) == static_cast<unsigned int>(uint1_d)) {}
  if (uint1_b.x == uint1_d.x) {}
  // CHECK: unsigned int uint1_h[16];
  uint1 uint1_h[16];
  // CHECK: unsigned int uint1_i[32];
  uint1 uint1_i[32];
  // CHECK: if (static_cast<unsigned int>(uint1_h[12]) == static_cast<unsigned int>(uint1_i[12])) {}
  if (uint1_h[12].x == uint1_i[12].x) {}
  // CHECK: uint1_f = (unsigned int *)uint1_i;
  uint1_f = (uint1 *)uint1_i;
  // CHECK: uint1_a = (unsigned int)uint1_c;
  uint1_a = (uint1)uint1_c;
  // CHECK: uint1_b = unsigned int(uint1_b);
  uint1_b = uint1(uint1_b);
  // CHECK: unsigned int uint1_j, uint1_k, uint1_l, uint1_m[16], *uint1_n[32];
  uint1 uint1_j, uint1_k, uint1_l, uint1_m[16], *uint1_n[32];
  // CHECK: int uint1_o = sizeof(unsigned int);
  int uint1_o = sizeof(uint1);
  // CHECK: int unsigned int_p = sizeof(unsigned int);
  int unsigned int_p = sizeof(unsigned int);
  // CHECK: int uint1_q = sizeof(uint1_d);
  int uint1_q = sizeof(uint1_d);
  int *uint1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uint1_e_buf_ct0 = dpct::get_buffer_and_offset(uint1_e);
  // CHECK-NEXT:   size_t uint1_e_offset_ct0 = uint1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uint1_cast_buf_ct1 = dpct::get_buffer_and_offset((unsigned int *)uint1_cast);
  // CHECK-NEXT:   size_t uint1_cast_offset_ct1 = uint1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto uint1_e_acc_ct0 = uint1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto uint1_cast_acc_ct1 = uint1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_uint1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           unsigned int *uint1_e_ct0 = (unsigned int *)(&uint1_e_acc_ct0[0] + uint1_e_offset_ct0);
  // CHECK-NEXT:           unsigned int *uint1_cast_ct1 = (unsigned int *)(&uint1_cast_acc_ct1[0] + uint1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_uint1(uint1_e_ct0, uint1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_uint1<<<1,1>>>(uint1_e, (uint1 *)uint1_cast);
  return 0;
}

// CHECK: void func3_uint2(cl::sycl::uint2 a, cl::sycl::uint2 b, cl::sycl::uint2 c) {
void func3_uint2(uint2 a, uint2 b, uint2 c) {
}
// CHECK: void func_uint2(cl::sycl::uint2 a) {
void func_uint2(uint2 a) {
}
// CHECK: void kernel_uint2(cl::sycl::uint2 *a, cl::sycl::uint2 *b) {
__global__ void kernel_uint2(uint2 *a, uint2 *b) {
}

int main_uint2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uint2 uint2_a;
  uint2 uint2_a;
  // CHECK: cl::sycl::uint2 uint2_b = cl::sycl::uint2(1, 2);
  uint2 uint2_b = make_uint2(1, 2);
  // CHECK: cl::sycl::uint2 uint2_c = cl::sycl::uint2(uint2_b);
  uint2 uint2_c = uint2(uint2_b);
  // CHECK: cl::sycl::uint2 uint2_d(uint2_c);
  uint2 uint2_d(uint2_c);
  // CHECK: func3_uint2(uint2_b, cl::sycl::uint2(uint2_b), (cl::sycl::uint2)uint2_b);
  func3_uint2(uint2_b, uint2(uint2_b), (uint2)uint2_b);
  // CHECK: cl::sycl::uint2 *uint2_e;
  uint2 *uint2_e;
  // CHECK: cl::sycl::uint2 *uint2_f;
  uint2 *uint2_f;
  // CHECK: unsigned int uint2_g = static_cast<unsigned int>(uint2_c.x());
  unsigned int uint2_g = uint2_c.x;
  // CHECK: uint2_a.x() = static_cast<unsigned int>(uint2_d.x());
  uint2_a.x = uint2_d.x;
  // CHECK: if (static_cast<unsigned int>(uint2_b.x()) == static_cast<unsigned int>(uint2_d.x())) {}
  if (uint2_b.x == uint2_d.x) {}
  // CHECK: cl::sycl::uint2 uint2_h[16];
  uint2 uint2_h[16];
  // CHECK: cl::sycl::uint2 uint2_i[32];
  uint2 uint2_i[32];
  // CHECK: if (static_cast<unsigned int>(uint2_h[12].x()) == static_cast<unsigned int>(uint2_i[12].x())) {}
  if (uint2_h[12].x == uint2_i[12].x) {}
  // CHECK: uint2_f = (cl::sycl::uint2 *)uint2_i;
  uint2_f = (uint2 *)uint2_i;
  // CHECK: uint2_a = (cl::sycl::uint2)uint2_c;
  uint2_a = (uint2)uint2_c;
  // CHECK: uint2_b = cl::sycl::uint2(uint2_b);
  uint2_b = uint2(uint2_b);
  // CHECK: cl::sycl::uint2 uint2_j, uint2_k, uint2_l, uint2_m[16], *uint2_n[32];
  uint2 uint2_j, uint2_k, uint2_l, uint2_m[16], *uint2_n[32];
  // CHECK: int uint2_o = sizeof(cl::sycl::uint2);
  int uint2_o = sizeof(uint2);
  // CHECK: int unsigned int_p = sizeof(unsigned int);
  int unsigned int_p = sizeof(unsigned int);
  // CHECK: int uint2_q = sizeof(uint2_d);
  int uint2_q = sizeof(uint2_d);
  int *uint2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uint2_e_buf_ct0 = dpct::get_buffer_and_offset(uint2_e);
  // CHECK-NEXT:   size_t uint2_e_offset_ct0 = uint2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uint2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::uint2 *)uint2_cast);
  // CHECK-NEXT:   size_t uint2_cast_offset_ct1 = uint2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto uint2_e_acc_ct0 = uint2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto uint2_cast_acc_ct1 = uint2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_uint2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::uint2 *uint2_e_ct0 = (cl::sycl::uint2 *)(&uint2_e_acc_ct0[0] + uint2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::uint2 *uint2_cast_ct1 = (cl::sycl::uint2 *)(&uint2_cast_acc_ct1[0] + uint2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_uint2(uint2_e_ct0, uint2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_uint2<<<1,1>>>(uint2_e, (uint2 *)uint2_cast);
  return 0;
}

// CHECK: void func3_uint3(cl::sycl::uint3 a, cl::sycl::uint3 b, cl::sycl::uint3 c) {
void func3_uint3(uint3 a, uint3 b, uint3 c) {
}
// CHECK: void func_uint3(cl::sycl::uint3 a) {
void func_uint3(uint3 a) {
}
// CHECK: void kernel_uint3(cl::sycl::uint3 *a, cl::sycl::uint3 *b) {
__global__ void kernel_uint3(uint3 *a, uint3 *b) {
}

int main_uint3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uint3 uint3_a;
  uint3 uint3_a;
  // CHECK: cl::sycl::uint3 uint3_b = cl::sycl::uint3(1, 2, 3);
  uint3 uint3_b = make_uint3(1, 2, 3);
  // CHECK: cl::sycl::uint3 uint3_c = cl::sycl::uint3(uint3_b);
  uint3 uint3_c = uint3(uint3_b);
  // CHECK: cl::sycl::uint3 uint3_d(uint3_c);
  uint3 uint3_d(uint3_c);
  // CHECK: func3_uint3(uint3_b, cl::sycl::uint3(uint3_b), (cl::sycl::uint3)uint3_b);
  func3_uint3(uint3_b, uint3(uint3_b), (uint3)uint3_b);
  // CHECK: cl::sycl::uint3 *uint3_e;
  uint3 *uint3_e;
  // CHECK: cl::sycl::uint3 *uint3_f;
  uint3 *uint3_f;
  // CHECK: unsigned int uint3_g = static_cast<unsigned int>(uint3_c.x());
  unsigned int uint3_g = uint3_c.x;
  // CHECK: uint3_a.x() = static_cast<unsigned int>(uint3_d.x());
  uint3_a.x = uint3_d.x;
  // CHECK: if (static_cast<unsigned int>(uint3_b.x()) == static_cast<unsigned int>(uint3_d.x())) {}
  if (uint3_b.x == uint3_d.x) {}
  // CHECK: cl::sycl::uint3 uint3_h[16];
  uint3 uint3_h[16];
  // CHECK: cl::sycl::uint3 uint3_i[32];
  uint3 uint3_i[32];
  // CHECK: if (static_cast<unsigned int>(uint3_h[12].x()) == static_cast<unsigned int>(uint3_i[12].x())) {}
  if (uint3_h[12].x == uint3_i[12].x) {}
  // CHECK: uint3_f = (cl::sycl::uint3 *)uint3_i;
  uint3_f = (uint3 *)uint3_i;
  // CHECK: uint3_a = (cl::sycl::uint3)uint3_c;
  uint3_a = (uint3)uint3_c;
  // CHECK: uint3_b = cl::sycl::uint3(uint3_b);
  uint3_b = uint3(uint3_b);
  // CHECK: cl::sycl::uint3 uint3_j, uint3_k, uint3_l, uint3_m[16], *uint3_n[32];
  uint3 uint3_j, uint3_k, uint3_l, uint3_m[16], *uint3_n[32];
  // CHECK: int uint3_o = sizeof(cl::sycl::uint3);
  int uint3_o = sizeof(uint3);
  // CHECK: int unsigned int_p = sizeof(unsigned int);
  int unsigned int_p = sizeof(unsigned int);
  // CHECK: int uint3_q = sizeof(uint3_d);
  int uint3_q = sizeof(uint3_d);
  int *uint3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uint3_e_buf_ct0 = dpct::get_buffer_and_offset(uint3_e);
  // CHECK-NEXT:   size_t uint3_e_offset_ct0 = uint3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uint3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::uint3 *)uint3_cast);
  // CHECK-NEXT:   size_t uint3_cast_offset_ct1 = uint3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto uint3_e_acc_ct0 = uint3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto uint3_cast_acc_ct1 = uint3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_uint3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::uint3 *uint3_e_ct0 = (cl::sycl::uint3 *)(&uint3_e_acc_ct0[0] + uint3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::uint3 *uint3_cast_ct1 = (cl::sycl::uint3 *)(&uint3_cast_acc_ct1[0] + uint3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_uint3(uint3_e_ct0, uint3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_uint3<<<1,1>>>(uint3_e, (uint3 *)uint3_cast);
  return 0;
}

// CHECK: void func3_uint4(cl::sycl::uint4 a, cl::sycl::uint4 b, cl::sycl::uint4 c) {
void func3_uint4(uint4 a, uint4 b, uint4 c) {
}
// CHECK: void func_uint4(cl::sycl::uint4 a) {
void func_uint4(uint4 a) {
}
// CHECK: void kernel_uint4(cl::sycl::uint4 *a, cl::sycl::uint4 *b) {
__global__ void kernel_uint4(uint4 *a, uint4 *b) {
}

int main_uint4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uint4 uint4_a;
  uint4 uint4_a;
  // CHECK: cl::sycl::uint4 uint4_b = cl::sycl::uint4(1, 2, 3, 4);
  uint4 uint4_b = make_uint4(1, 2, 3, 4);
  // CHECK: cl::sycl::uint4 uint4_c = cl::sycl::uint4(uint4_b);
  uint4 uint4_c = uint4(uint4_b);
  // CHECK: cl::sycl::uint4 uint4_d(uint4_c);
  uint4 uint4_d(uint4_c);
  // CHECK: func3_uint4(uint4_b, cl::sycl::uint4(uint4_b), (cl::sycl::uint4)uint4_b);
  func3_uint4(uint4_b, uint4(uint4_b), (uint4)uint4_b);
  // CHECK: cl::sycl::uint4 *uint4_e;
  uint4 *uint4_e;
  // CHECK: cl::sycl::uint4 *uint4_f;
  uint4 *uint4_f;
  // CHECK: unsigned int uint4_g = static_cast<unsigned int>(uint4_c.x());
  unsigned int uint4_g = uint4_c.x;
  // CHECK: uint4_a.x() = static_cast<unsigned int>(uint4_d.x());
  uint4_a.x = uint4_d.x;
  // CHECK: if (static_cast<unsigned int>(uint4_b.x()) == static_cast<unsigned int>(uint4_d.x())) {}
  if (uint4_b.x == uint4_d.x) {}
  // CHECK: cl::sycl::uint4 uint4_h[16];
  uint4 uint4_h[16];
  // CHECK: cl::sycl::uint4 uint4_i[32];
  uint4 uint4_i[32];
  // CHECK: if (static_cast<unsigned int>(uint4_h[12].x()) == static_cast<unsigned int>(uint4_i[12].x())) {}
  if (uint4_h[12].x == uint4_i[12].x) {}
  // CHECK: uint4_f = (cl::sycl::uint4 *)uint4_i;
  uint4_f = (uint4 *)uint4_i;
  // CHECK: uint4_a = (cl::sycl::uint4)uint4_c;
  uint4_a = (uint4)uint4_c;
  // CHECK: uint4_b = cl::sycl::uint4(uint4_b);
  uint4_b = uint4(uint4_b);
  // CHECK: cl::sycl::uint4 uint4_j, uint4_k, uint4_l, uint4_m[16], *uint4_n[32];
  uint4 uint4_j, uint4_k, uint4_l, uint4_m[16], *uint4_n[32];
  // CHECK: int uint4_o = sizeof(cl::sycl::uint4);
  int uint4_o = sizeof(uint4);
  // CHECK: int unsigned int_p = sizeof(unsigned int);
  int unsigned int_p = sizeof(unsigned int);
  // CHECK: int uint4_q = sizeof(uint4_d);
  int uint4_q = sizeof(uint4_d);
  int *uint4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uint4_e_buf_ct0 = dpct::get_buffer_and_offset(uint4_e);
  // CHECK-NEXT:   size_t uint4_e_offset_ct0 = uint4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> uint4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::uint4 *)uint4_cast);
  // CHECK-NEXT:   size_t uint4_cast_offset_ct1 = uint4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto uint4_e_acc_ct0 = uint4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto uint4_cast_acc_ct1 = uint4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_uint4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::uint4 *uint4_e_ct0 = (cl::sycl::uint4 *)(&uint4_e_acc_ct0[0] + uint4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::uint4 *uint4_cast_ct1 = (cl::sycl::uint4 *)(&uint4_cast_acc_ct1[0] + uint4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_uint4(uint4_e_ct0, uint4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_uint4<<<1,1>>>(uint4_e, (uint4 *)uint4_cast);
  return 0;
}

// CHECK: void func3_ulong1(unsigned long a, unsigned long b, unsigned long c) {
void func3_ulong1(ulong1 a, ulong1 b, ulong1 c) {
}
// CHECK: void func_ulong1(unsigned long a) {
void func_ulong1(ulong1 a) {
}
// CHECK: void kernel_ulong1(unsigned long *a, unsigned long *b) {
__global__ void kernel_ulong1(ulong1 *a, ulong1 *b) {
}

int main_ulong1() {
  // range default constructor does the right thing.
  // CHECK: unsigned long ulong1_a;
  ulong1 ulong1_a;
  // CHECK: unsigned long ulong1_b = unsigned long(1);
  ulong1 ulong1_b = make_ulong1(1);
  // CHECK: unsigned long ulong1_c = unsigned long(ulong1_b);
  ulong1 ulong1_c = ulong1(ulong1_b);
  // CHECK: unsigned long ulong1_d(ulong1_c);
  ulong1 ulong1_d(ulong1_c);
  // CHECK: func3_ulong1(ulong1_b, unsigned long(ulong1_b), (unsigned long)ulong1_b);
  func3_ulong1(ulong1_b, ulong1(ulong1_b), (ulong1)ulong1_b);
  // CHECK: unsigned long *ulong1_e;
  ulong1 *ulong1_e;
  // CHECK: unsigned long *ulong1_f;
  ulong1 *ulong1_f;
  // CHECK: unsigned long ulong1_g = static_cast<unsigned long>(ulong1_c);
  unsigned long ulong1_g = ulong1_c.x;
  // CHECK: ulong1_a = static_cast<unsigned long>(ulong1_d);
  ulong1_a.x = ulong1_d.x;
  // CHECK: if (static_cast<unsigned long>(ulong1_b) == static_cast<unsigned long>(ulong1_d)) {}
  if (ulong1_b.x == ulong1_d.x) {}
  // CHECK: unsigned long ulong1_h[16];
  ulong1 ulong1_h[16];
  // CHECK: unsigned long ulong1_i[32];
  ulong1 ulong1_i[32];
  // CHECK: if (static_cast<unsigned long>(ulong1_h[12]) == static_cast<unsigned long>(ulong1_i[12])) {}
  if (ulong1_h[12].x == ulong1_i[12].x) {}
  // CHECK: ulong1_f = (unsigned long *)ulong1_i;
  ulong1_f = (ulong1 *)ulong1_i;
  // CHECK: ulong1_a = (unsigned long)ulong1_c;
  ulong1_a = (ulong1)ulong1_c;
  // CHECK: ulong1_b = unsigned long(ulong1_b);
  ulong1_b = ulong1(ulong1_b);
  // CHECK: unsigned long ulong1_j, ulong1_k, ulong1_l, ulong1_m[16], *ulong1_n[32];
  ulong1 ulong1_j, ulong1_k, ulong1_l, ulong1_m[16], *ulong1_n[32];
  // CHECK: int ulong1_o = sizeof(unsigned long);
  int ulong1_o = sizeof(ulong1);
  // CHECK: int unsigned long_p = sizeof(unsigned long);
  int unsigned long_p = sizeof(unsigned long);
  // CHECK: int ulong1_q = sizeof(ulong1_d);
  int ulong1_q = sizeof(ulong1_d);
  int *ulong1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulong1_e_buf_ct0 = dpct::get_buffer_and_offset(ulong1_e);
  // CHECK-NEXT:   size_t ulong1_e_offset_ct0 = ulong1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulong1_cast_buf_ct1 = dpct::get_buffer_and_offset((unsigned long *)ulong1_cast);
  // CHECK-NEXT:   size_t ulong1_cast_offset_ct1 = ulong1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ulong1_e_acc_ct0 = ulong1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ulong1_cast_acc_ct1 = ulong1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ulong1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           unsigned long *ulong1_e_ct0 = (unsigned long *)(&ulong1_e_acc_ct0[0] + ulong1_e_offset_ct0);
  // CHECK-NEXT:           unsigned long *ulong1_cast_ct1 = (unsigned long *)(&ulong1_cast_acc_ct1[0] + ulong1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ulong1(ulong1_e_ct0, ulong1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ulong1<<<1,1>>>(ulong1_e, (ulong1 *)ulong1_cast);
  return 0;
}

// CHECK: void func3_ulong2(cl::sycl::ulong2 a, cl::sycl::ulong2 b, cl::sycl::ulong2 c) {
void func3_ulong2(ulong2 a, ulong2 b, ulong2 c) {
}
// CHECK: void func_ulong2(cl::sycl::ulong2 a) {
void func_ulong2(ulong2 a) {
}
// CHECK: void kernel_ulong2(cl::sycl::ulong2 *a, cl::sycl::ulong2 *b) {
__global__ void kernel_ulong2(ulong2 *a, ulong2 *b) {
}

int main_ulong2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulong2 ulong2_a;
  ulong2 ulong2_a;
  // CHECK: cl::sycl::ulong2 ulong2_b = cl::sycl::ulong2(1, 2);
  ulong2 ulong2_b = make_ulong2(1, 2);
  // CHECK: cl::sycl::ulong2 ulong2_c = cl::sycl::ulong2(ulong2_b);
  ulong2 ulong2_c = ulong2(ulong2_b);
  // CHECK: cl::sycl::ulong2 ulong2_d(ulong2_c);
  ulong2 ulong2_d(ulong2_c);
  // CHECK: func3_ulong2(ulong2_b, cl::sycl::ulong2(ulong2_b), (cl::sycl::ulong2)ulong2_b);
  func3_ulong2(ulong2_b, ulong2(ulong2_b), (ulong2)ulong2_b);
  // CHECK: cl::sycl::ulong2 *ulong2_e;
  ulong2 *ulong2_e;
  // CHECK: cl::sycl::ulong2 *ulong2_f;
  ulong2 *ulong2_f;
  // CHECK: unsigned long ulong2_g = static_cast<unsigned long>(ulong2_c.x());
  unsigned long ulong2_g = ulong2_c.x;
  // CHECK: ulong2_a.x() = static_cast<unsigned long>(ulong2_d.x());
  ulong2_a.x = ulong2_d.x;
  // CHECK: if (static_cast<unsigned long>(ulong2_b.x()) == static_cast<unsigned long>(ulong2_d.x())) {}
  if (ulong2_b.x == ulong2_d.x) {}
  // CHECK: cl::sycl::ulong2 ulong2_h[16];
  ulong2 ulong2_h[16];
  // CHECK: cl::sycl::ulong2 ulong2_i[32];
  ulong2 ulong2_i[32];
  // CHECK: if (static_cast<unsigned long>(ulong2_h[12].x()) == static_cast<unsigned long>(ulong2_i[12].x())) {}
  if (ulong2_h[12].x == ulong2_i[12].x) {}
  // CHECK: ulong2_f = (cl::sycl::ulong2 *)ulong2_i;
  ulong2_f = (ulong2 *)ulong2_i;
  // CHECK: ulong2_a = (cl::sycl::ulong2)ulong2_c;
  ulong2_a = (ulong2)ulong2_c;
  // CHECK: ulong2_b = cl::sycl::ulong2(ulong2_b);
  ulong2_b = ulong2(ulong2_b);
  // CHECK: cl::sycl::ulong2 ulong2_j, ulong2_k, ulong2_l, ulong2_m[16], *ulong2_n[32];
  ulong2 ulong2_j, ulong2_k, ulong2_l, ulong2_m[16], *ulong2_n[32];
  // CHECK: int ulong2_o = sizeof(cl::sycl::ulong2);
  int ulong2_o = sizeof(ulong2);
  // CHECK: int unsigned long_p = sizeof(unsigned long);
  int unsigned long_p = sizeof(unsigned long);
  // CHECK: int ulong2_q = sizeof(ulong2_d);
  int ulong2_q = sizeof(ulong2_d);
  int *ulong2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulong2_e_buf_ct0 = dpct::get_buffer_and_offset(ulong2_e);
  // CHECK-NEXT:   size_t ulong2_e_offset_ct0 = ulong2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulong2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::ulong2 *)ulong2_cast);
  // CHECK-NEXT:   size_t ulong2_cast_offset_ct1 = ulong2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ulong2_e_acc_ct0 = ulong2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ulong2_cast_acc_ct1 = ulong2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ulong2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::ulong2 *ulong2_e_ct0 = (cl::sycl::ulong2 *)(&ulong2_e_acc_ct0[0] + ulong2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::ulong2 *ulong2_cast_ct1 = (cl::sycl::ulong2 *)(&ulong2_cast_acc_ct1[0] + ulong2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ulong2(ulong2_e_ct0, ulong2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ulong2<<<1,1>>>(ulong2_e, (ulong2 *)ulong2_cast);
  return 0;
}

// CHECK: void func3_ulong3(cl::sycl::ulong3 a, cl::sycl::ulong3 b, cl::sycl::ulong3 c) {
void func3_ulong3(ulong3 a, ulong3 b, ulong3 c) {
}
// CHECK: void func_ulong3(cl::sycl::ulong3 a) {
void func_ulong3(ulong3 a) {
}
// CHECK: void kernel_ulong3(cl::sycl::ulong3 *a, cl::sycl::ulong3 *b) {
__global__ void kernel_ulong3(ulong3 *a, ulong3 *b) {
}

int main_ulong3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulong3 ulong3_a;
  ulong3 ulong3_a;
  // CHECK: cl::sycl::ulong3 ulong3_b = cl::sycl::ulong3(1, 2, 3);
  ulong3 ulong3_b = make_ulong3(1, 2, 3);
  // CHECK: cl::sycl::ulong3 ulong3_c = cl::sycl::ulong3(ulong3_b);
  ulong3 ulong3_c = ulong3(ulong3_b);
  // CHECK: cl::sycl::ulong3 ulong3_d(ulong3_c);
  ulong3 ulong3_d(ulong3_c);
  // CHECK: func3_ulong3(ulong3_b, cl::sycl::ulong3(ulong3_b), (cl::sycl::ulong3)ulong3_b);
  func3_ulong3(ulong3_b, ulong3(ulong3_b), (ulong3)ulong3_b);
  // CHECK: cl::sycl::ulong3 *ulong3_e;
  ulong3 *ulong3_e;
  // CHECK: cl::sycl::ulong3 *ulong3_f;
  ulong3 *ulong3_f;
  // CHECK: unsigned long ulong3_g = static_cast<unsigned long>(ulong3_c.x());
  unsigned long ulong3_g = ulong3_c.x;
  // CHECK: ulong3_a.x() = static_cast<unsigned long>(ulong3_d.x());
  ulong3_a.x = ulong3_d.x;
  // CHECK: if (static_cast<unsigned long>(ulong3_b.x()) == static_cast<unsigned long>(ulong3_d.x())) {}
  if (ulong3_b.x == ulong3_d.x) {}
  // CHECK: cl::sycl::ulong3 ulong3_h[16];
  ulong3 ulong3_h[16];
  // CHECK: cl::sycl::ulong3 ulong3_i[32];
  ulong3 ulong3_i[32];
  // CHECK: if (static_cast<unsigned long>(ulong3_h[12].x()) == static_cast<unsigned long>(ulong3_i[12].x())) {}
  if (ulong3_h[12].x == ulong3_i[12].x) {}
  // CHECK: ulong3_f = (cl::sycl::ulong3 *)ulong3_i;
  ulong3_f = (ulong3 *)ulong3_i;
  // CHECK: ulong3_a = (cl::sycl::ulong3)ulong3_c;
  ulong3_a = (ulong3)ulong3_c;
  // CHECK: ulong3_b = cl::sycl::ulong3(ulong3_b);
  ulong3_b = ulong3(ulong3_b);
  // CHECK: cl::sycl::ulong3 ulong3_j, ulong3_k, ulong3_l, ulong3_m[16], *ulong3_n[32];
  ulong3 ulong3_j, ulong3_k, ulong3_l, ulong3_m[16], *ulong3_n[32];
  // CHECK: int ulong3_o = sizeof(cl::sycl::ulong3);
  int ulong3_o = sizeof(ulong3);
  // CHECK: int unsigned long_p = sizeof(unsigned long);
  int unsigned long_p = sizeof(unsigned long);
  // CHECK: int ulong3_q = sizeof(ulong3_d);
  int ulong3_q = sizeof(ulong3_d);
  int *ulong3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulong3_e_buf_ct0 = dpct::get_buffer_and_offset(ulong3_e);
  // CHECK-NEXT:   size_t ulong3_e_offset_ct0 = ulong3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulong3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::ulong3 *)ulong3_cast);
  // CHECK-NEXT:   size_t ulong3_cast_offset_ct1 = ulong3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ulong3_e_acc_ct0 = ulong3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ulong3_cast_acc_ct1 = ulong3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ulong3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::ulong3 *ulong3_e_ct0 = (cl::sycl::ulong3 *)(&ulong3_e_acc_ct0[0] + ulong3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::ulong3 *ulong3_cast_ct1 = (cl::sycl::ulong3 *)(&ulong3_cast_acc_ct1[0] + ulong3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ulong3(ulong3_e_ct0, ulong3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ulong3<<<1,1>>>(ulong3_e, (ulong3 *)ulong3_cast);
  return 0;
}

// CHECK: void func3_ulong4(cl::sycl::ulong4 a, cl::sycl::ulong4 b, cl::sycl::ulong4 c) {
void func3_ulong4(ulong4 a, ulong4 b, ulong4 c) {
}
// CHECK: void func_ulong4(cl::sycl::ulong4 a) {
void func_ulong4(ulong4 a) {
}
// CHECK: void kernel_ulong4(cl::sycl::ulong4 *a, cl::sycl::ulong4 *b) {
__global__ void kernel_ulong4(ulong4 *a, ulong4 *b) {
}

int main_ulong4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulong4 ulong4_a;
  ulong4 ulong4_a;
  // CHECK: cl::sycl::ulong4 ulong4_b = cl::sycl::ulong4(1, 2, 3, 4);
  ulong4 ulong4_b = make_ulong4(1, 2, 3, 4);
  // CHECK: cl::sycl::ulong4 ulong4_c = cl::sycl::ulong4(ulong4_b);
  ulong4 ulong4_c = ulong4(ulong4_b);
  // CHECK: cl::sycl::ulong4 ulong4_d(ulong4_c);
  ulong4 ulong4_d(ulong4_c);
  // CHECK: func3_ulong4(ulong4_b, cl::sycl::ulong4(ulong4_b), (cl::sycl::ulong4)ulong4_b);
  func3_ulong4(ulong4_b, ulong4(ulong4_b), (ulong4)ulong4_b);
  // CHECK: cl::sycl::ulong4 *ulong4_e;
  ulong4 *ulong4_e;
  // CHECK: cl::sycl::ulong4 *ulong4_f;
  ulong4 *ulong4_f;
  // CHECK: unsigned long ulong4_g = static_cast<unsigned long>(ulong4_c.x());
  unsigned long ulong4_g = ulong4_c.x;
  // CHECK: ulong4_a.x() = static_cast<unsigned long>(ulong4_d.x());
  ulong4_a.x = ulong4_d.x;
  // CHECK: if (static_cast<unsigned long>(ulong4_b.x()) == static_cast<unsigned long>(ulong4_d.x())) {}
  if (ulong4_b.x == ulong4_d.x) {}
  // CHECK: cl::sycl::ulong4 ulong4_h[16];
  ulong4 ulong4_h[16];
  // CHECK: cl::sycl::ulong4 ulong4_i[32];
  ulong4 ulong4_i[32];
  // CHECK: if (static_cast<unsigned long>(ulong4_h[12].x()) == static_cast<unsigned long>(ulong4_i[12].x())) {}
  if (ulong4_h[12].x == ulong4_i[12].x) {}
  // CHECK: ulong4_f = (cl::sycl::ulong4 *)ulong4_i;
  ulong4_f = (ulong4 *)ulong4_i;
  // CHECK: ulong4_a = (cl::sycl::ulong4)ulong4_c;
  ulong4_a = (ulong4)ulong4_c;
  // CHECK: ulong4_b = cl::sycl::ulong4(ulong4_b);
  ulong4_b = ulong4(ulong4_b);
  // CHECK: cl::sycl::ulong4 ulong4_j, ulong4_k, ulong4_l, ulong4_m[16], *ulong4_n[32];
  ulong4 ulong4_j, ulong4_k, ulong4_l, ulong4_m[16], *ulong4_n[32];
  // CHECK: int ulong4_o = sizeof(cl::sycl::ulong4);
  int ulong4_o = sizeof(ulong4);
  // CHECK: int unsigned long_p = sizeof(unsigned long);
  int unsigned long_p = sizeof(unsigned long);
  // CHECK: int ulong4_q = sizeof(ulong4_d);
  int ulong4_q = sizeof(ulong4_d);
  int *ulong4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulong4_e_buf_ct0 = dpct::get_buffer_and_offset(ulong4_e);
  // CHECK-NEXT:   size_t ulong4_e_offset_ct0 = ulong4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulong4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::ulong4 *)ulong4_cast);
  // CHECK-NEXT:   size_t ulong4_cast_offset_ct1 = ulong4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ulong4_e_acc_ct0 = ulong4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ulong4_cast_acc_ct1 = ulong4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ulong4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::ulong4 *ulong4_e_ct0 = (cl::sycl::ulong4 *)(&ulong4_e_acc_ct0[0] + ulong4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::ulong4 *ulong4_cast_ct1 = (cl::sycl::ulong4 *)(&ulong4_cast_acc_ct1[0] + ulong4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ulong4(ulong4_e_ct0, ulong4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ulong4<<<1,1>>>(ulong4_e, (ulong4 *)ulong4_cast);
  return 0;
}

// CHECK: void func3_ulonglong1(unsigned long long a, unsigned long long b, unsigned long long c) {
void func3_ulonglong1(ulonglong1 a, ulonglong1 b, ulonglong1 c) {
}
// CHECK: void func_ulonglong1(unsigned long long a) {
void func_ulonglong1(ulonglong1 a) {
}
// CHECK: void kernel_ulonglong1(unsigned long long *a, unsigned long long *b) {
__global__ void kernel_ulonglong1(ulonglong1 *a, ulonglong1 *b) {
}

int main_ulonglong1() {
  // range default constructor does the right thing.
  // CHECK: unsigned long long ulonglong1_a;
  ulonglong1 ulonglong1_a;
  // CHECK: unsigned long long ulonglong1_b = unsigned long long(1);
  ulonglong1 ulonglong1_b = make_ulonglong1(1);
  // CHECK: unsigned long long ulonglong1_c = unsigned long long(ulonglong1_b);
  ulonglong1 ulonglong1_c = ulonglong1(ulonglong1_b);
  // CHECK: unsigned long long ulonglong1_d(ulonglong1_c);
  ulonglong1 ulonglong1_d(ulonglong1_c);
  // CHECK: func3_ulonglong1(ulonglong1_b, unsigned long long(ulonglong1_b), (unsigned long long)ulonglong1_b);
  func3_ulonglong1(ulonglong1_b, ulonglong1(ulonglong1_b), (ulonglong1)ulonglong1_b);
  // CHECK: unsigned long long *ulonglong1_e;
  ulonglong1 *ulonglong1_e;
  // CHECK: unsigned long long *ulonglong1_f;
  ulonglong1 *ulonglong1_f;
  // CHECK: unsigned long long ulonglong1_g = static_cast<unsigned long long>(ulonglong1_c);
  unsigned long long ulonglong1_g = ulonglong1_c.x;
  // CHECK: ulonglong1_a = static_cast<unsigned long long>(ulonglong1_d);
  ulonglong1_a.x = ulonglong1_d.x;
  // CHECK: if (static_cast<unsigned long long>(ulonglong1_b) == static_cast<unsigned long long>(ulonglong1_d)) {}
  if (ulonglong1_b.x == ulonglong1_d.x) {}
  // CHECK: unsigned long long ulonglong1_h[16];
  ulonglong1 ulonglong1_h[16];
  // CHECK: unsigned long long ulonglong1_i[32];
  ulonglong1 ulonglong1_i[32];
  // CHECK: if (static_cast<unsigned long long>(ulonglong1_h[12]) == static_cast<unsigned long long>(ulonglong1_i[12])) {}
  if (ulonglong1_h[12].x == ulonglong1_i[12].x) {}
  // CHECK: ulonglong1_f = (unsigned long long *)ulonglong1_i;
  ulonglong1_f = (ulonglong1 *)ulonglong1_i;
  // CHECK: ulonglong1_a = (unsigned long long)ulonglong1_c;
  ulonglong1_a = (ulonglong1)ulonglong1_c;
  // CHECK: ulonglong1_b = unsigned long long(ulonglong1_b);
  ulonglong1_b = ulonglong1(ulonglong1_b);
  // CHECK: unsigned long long ulonglong1_j, ulonglong1_k, ulonglong1_l, ulonglong1_m[16], *ulonglong1_n[32];
  ulonglong1 ulonglong1_j, ulonglong1_k, ulonglong1_l, ulonglong1_m[16], *ulonglong1_n[32];
  // CHECK: int ulonglong1_o = sizeof(unsigned long long);
  int ulonglong1_o = sizeof(ulonglong1);
  // CHECK: int unsigned long long_p = sizeof(unsigned long long);
  int unsigned long long_p = sizeof(unsigned long long);
  // CHECK: int ulonglong1_q = sizeof(ulonglong1_d);
  int ulonglong1_q = sizeof(ulonglong1_d);
  int *ulonglong1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulonglong1_e_buf_ct0 = dpct::get_buffer_and_offset(ulonglong1_e);
  // CHECK-NEXT:   size_t ulonglong1_e_offset_ct0 = ulonglong1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulonglong1_cast_buf_ct1 = dpct::get_buffer_and_offset((unsigned long long *)ulonglong1_cast);
  // CHECK-NEXT:   size_t ulonglong1_cast_offset_ct1 = ulonglong1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ulonglong1_e_acc_ct0 = ulonglong1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ulonglong1_cast_acc_ct1 = ulonglong1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ulonglong1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           unsigned long long *ulonglong1_e_ct0 = (unsigned long long *)(&ulonglong1_e_acc_ct0[0] + ulonglong1_e_offset_ct0);
  // CHECK-NEXT:           unsigned long long *ulonglong1_cast_ct1 = (unsigned long long *)(&ulonglong1_cast_acc_ct1[0] + ulonglong1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ulonglong1(ulonglong1_e_ct0, ulonglong1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ulonglong1<<<1,1>>>(ulonglong1_e, (ulonglong1 *)ulonglong1_cast);
  return 0;
}

// CHECK: void func3_ulonglong2(cl::sycl::ulonglong2 a, cl::sycl::ulonglong2 b, cl::sycl::ulonglong2 c) {
void func3_ulonglong2(ulonglong2 a, ulonglong2 b, ulonglong2 c) {
}
// CHECK: void func_ulonglong2(cl::sycl::ulonglong2 a) {
void func_ulonglong2(ulonglong2 a) {
}
// CHECK: void kernel_ulonglong2(cl::sycl::ulonglong2 *a, cl::sycl::ulonglong2 *b) {
__global__ void kernel_ulonglong2(ulonglong2 *a, ulonglong2 *b) {
}

int main_ulonglong2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulonglong2 ulonglong2_a;
  ulonglong2 ulonglong2_a;
  // CHECK: cl::sycl::ulonglong2 ulonglong2_b = cl::sycl::ulonglong2(1, 2);
  ulonglong2 ulonglong2_b = make_ulonglong2(1, 2);
  // CHECK: cl::sycl::ulonglong2 ulonglong2_c = cl::sycl::ulonglong2(ulonglong2_b);
  ulonglong2 ulonglong2_c = ulonglong2(ulonglong2_b);
  // CHECK: cl::sycl::ulonglong2 ulonglong2_d(ulonglong2_c);
  ulonglong2 ulonglong2_d(ulonglong2_c);
  // CHECK: func3_ulonglong2(ulonglong2_b, cl::sycl::ulonglong2(ulonglong2_b), (cl::sycl::ulonglong2)ulonglong2_b);
  func3_ulonglong2(ulonglong2_b, ulonglong2(ulonglong2_b), (ulonglong2)ulonglong2_b);
  // CHECK: cl::sycl::ulonglong2 *ulonglong2_e;
  ulonglong2 *ulonglong2_e;
  // CHECK: cl::sycl::ulonglong2 *ulonglong2_f;
  ulonglong2 *ulonglong2_f;
  // CHECK: unsigned long long ulonglong2_g = static_cast<unsigned long long>(ulonglong2_c.x());
  unsigned long long ulonglong2_g = ulonglong2_c.x;
  // CHECK: ulonglong2_a.x() = static_cast<unsigned long long>(ulonglong2_d.x());
  ulonglong2_a.x = ulonglong2_d.x;
  // CHECK: if (static_cast<unsigned long long>(ulonglong2_b.x()) == static_cast<unsigned long long>(ulonglong2_d.x())) {}
  if (ulonglong2_b.x == ulonglong2_d.x) {}
  // CHECK: cl::sycl::ulonglong2 ulonglong2_h[16];
  ulonglong2 ulonglong2_h[16];
  // CHECK: cl::sycl::ulonglong2 ulonglong2_i[32];
  ulonglong2 ulonglong2_i[32];
  // CHECK: if (static_cast<unsigned long long>(ulonglong2_h[12].x()) == static_cast<unsigned long long>(ulonglong2_i[12].x())) {}
  if (ulonglong2_h[12].x == ulonglong2_i[12].x) {}
  // CHECK: ulonglong2_f = (cl::sycl::ulonglong2 *)ulonglong2_i;
  ulonglong2_f = (ulonglong2 *)ulonglong2_i;
  // CHECK: ulonglong2_a = (cl::sycl::ulonglong2)ulonglong2_c;
  ulonglong2_a = (ulonglong2)ulonglong2_c;
  // CHECK: ulonglong2_b = cl::sycl::ulonglong2(ulonglong2_b);
  ulonglong2_b = ulonglong2(ulonglong2_b);
  // CHECK: cl::sycl::ulonglong2 ulonglong2_j, ulonglong2_k, ulonglong2_l, ulonglong2_m[16], *ulonglong2_n[32];
  ulonglong2 ulonglong2_j, ulonglong2_k, ulonglong2_l, ulonglong2_m[16], *ulonglong2_n[32];
  // CHECK: int ulonglong2_o = sizeof(cl::sycl::ulonglong2);
  int ulonglong2_o = sizeof(ulonglong2);
  // CHECK: int unsigned long long_p = sizeof(unsigned long long);
  int unsigned long long_p = sizeof(unsigned long long);
  // CHECK: int ulonglong2_q = sizeof(ulonglong2_d);
  int ulonglong2_q = sizeof(ulonglong2_d);
  int *ulonglong2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulonglong2_e_buf_ct0 = dpct::get_buffer_and_offset(ulonglong2_e);
  // CHECK-NEXT:   size_t ulonglong2_e_offset_ct0 = ulonglong2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulonglong2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::ulonglong2 *)ulonglong2_cast);
  // CHECK-NEXT:   size_t ulonglong2_cast_offset_ct1 = ulonglong2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ulonglong2_e_acc_ct0 = ulonglong2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ulonglong2_cast_acc_ct1 = ulonglong2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ulonglong2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::ulonglong2 *ulonglong2_e_ct0 = (cl::sycl::ulonglong2 *)(&ulonglong2_e_acc_ct0[0] + ulonglong2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::ulonglong2 *ulonglong2_cast_ct1 = (cl::sycl::ulonglong2 *)(&ulonglong2_cast_acc_ct1[0] + ulonglong2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ulonglong2(ulonglong2_e_ct0, ulonglong2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ulonglong2<<<1,1>>>(ulonglong2_e, (ulonglong2 *)ulonglong2_cast);
  return 0;
}

// CHECK: void func3_ulonglong3(cl::sycl::ulonglong3 a, cl::sycl::ulonglong3 b, cl::sycl::ulonglong3 c) {
void func3_ulonglong3(ulonglong3 a, ulonglong3 b, ulonglong3 c) {
}
// CHECK: void func_ulonglong3(cl::sycl::ulonglong3 a) {
void func_ulonglong3(ulonglong3 a) {
}
// CHECK: void kernel_ulonglong3(cl::sycl::ulonglong3 *a, cl::sycl::ulonglong3 *b) {
__global__ void kernel_ulonglong3(ulonglong3 *a, ulonglong3 *b) {
}

int main_ulonglong3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulonglong3 ulonglong3_a;
  ulonglong3 ulonglong3_a;
  // CHECK: cl::sycl::ulonglong3 ulonglong3_b = cl::sycl::ulonglong3(1, 2, 3);
  ulonglong3 ulonglong3_b = make_ulonglong3(1, 2, 3);
  // CHECK: cl::sycl::ulonglong3 ulonglong3_c = cl::sycl::ulonglong3(ulonglong3_b);
  ulonglong3 ulonglong3_c = ulonglong3(ulonglong3_b);
  // CHECK: cl::sycl::ulonglong3 ulonglong3_d(ulonglong3_c);
  ulonglong3 ulonglong3_d(ulonglong3_c);
  // CHECK: func3_ulonglong3(ulonglong3_b, cl::sycl::ulonglong3(ulonglong3_b), (cl::sycl::ulonglong3)ulonglong3_b);
  func3_ulonglong3(ulonglong3_b, ulonglong3(ulonglong3_b), (ulonglong3)ulonglong3_b);
  // CHECK: cl::sycl::ulonglong3 *ulonglong3_e;
  ulonglong3 *ulonglong3_e;
  // CHECK: cl::sycl::ulonglong3 *ulonglong3_f;
  ulonglong3 *ulonglong3_f;
  // CHECK: unsigned long long ulonglong3_g = static_cast<unsigned long long>(ulonglong3_c.x());
  unsigned long long ulonglong3_g = ulonglong3_c.x;
  // CHECK: ulonglong3_a.x() = static_cast<unsigned long long>(ulonglong3_d.x());
  ulonglong3_a.x = ulonglong3_d.x;
  // CHECK: if (static_cast<unsigned long long>(ulonglong3_b.x()) == static_cast<unsigned long long>(ulonglong3_d.x())) {}
  if (ulonglong3_b.x == ulonglong3_d.x) {}
  // CHECK: cl::sycl::ulonglong3 ulonglong3_h[16];
  ulonglong3 ulonglong3_h[16];
  // CHECK: cl::sycl::ulonglong3 ulonglong3_i[32];
  ulonglong3 ulonglong3_i[32];
  // CHECK: if (static_cast<unsigned long long>(ulonglong3_h[12].x()) == static_cast<unsigned long long>(ulonglong3_i[12].x())) {}
  if (ulonglong3_h[12].x == ulonglong3_i[12].x) {}
  // CHECK: ulonglong3_f = (cl::sycl::ulonglong3 *)ulonglong3_i;
  ulonglong3_f = (ulonglong3 *)ulonglong3_i;
  // CHECK: ulonglong3_a = (cl::sycl::ulonglong3)ulonglong3_c;
  ulonglong3_a = (ulonglong3)ulonglong3_c;
  // CHECK: ulonglong3_b = cl::sycl::ulonglong3(ulonglong3_b);
  ulonglong3_b = ulonglong3(ulonglong3_b);
  // CHECK: cl::sycl::ulonglong3 ulonglong3_j, ulonglong3_k, ulonglong3_l, ulonglong3_m[16], *ulonglong3_n[32];
  ulonglong3 ulonglong3_j, ulonglong3_k, ulonglong3_l, ulonglong3_m[16], *ulonglong3_n[32];
  // CHECK: int ulonglong3_o = sizeof(cl::sycl::ulonglong3);
  int ulonglong3_o = sizeof(ulonglong3);
  // CHECK: int unsigned long long_p = sizeof(unsigned long long);
  int unsigned long long_p = sizeof(unsigned long long);
  // CHECK: int ulonglong3_q = sizeof(ulonglong3_d);
  int ulonglong3_q = sizeof(ulonglong3_d);
  int *ulonglong3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulonglong3_e_buf_ct0 = dpct::get_buffer_and_offset(ulonglong3_e);
  // CHECK-NEXT:   size_t ulonglong3_e_offset_ct0 = ulonglong3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulonglong3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::ulonglong3 *)ulonglong3_cast);
  // CHECK-NEXT:   size_t ulonglong3_cast_offset_ct1 = ulonglong3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ulonglong3_e_acc_ct0 = ulonglong3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ulonglong3_cast_acc_ct1 = ulonglong3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ulonglong3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::ulonglong3 *ulonglong3_e_ct0 = (cl::sycl::ulonglong3 *)(&ulonglong3_e_acc_ct0[0] + ulonglong3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::ulonglong3 *ulonglong3_cast_ct1 = (cl::sycl::ulonglong3 *)(&ulonglong3_cast_acc_ct1[0] + ulonglong3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ulonglong3(ulonglong3_e_ct0, ulonglong3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ulonglong3<<<1,1>>>(ulonglong3_e, (ulonglong3 *)ulonglong3_cast);
  return 0;
}

// CHECK: void func3_ulonglong4(cl::sycl::ulonglong4 a, cl::sycl::ulonglong4 b, cl::sycl::ulonglong4 c) {
void func3_ulonglong4(ulonglong4 a, ulonglong4 b, ulonglong4 c) {
}
// CHECK: void func_ulonglong4(cl::sycl::ulonglong4 a) {
void func_ulonglong4(ulonglong4 a) {
}
// CHECK: void kernel_ulonglong4(cl::sycl::ulonglong4 *a, cl::sycl::ulonglong4 *b) {
__global__ void kernel_ulonglong4(ulonglong4 *a, ulonglong4 *b) {
}

int main_ulonglong4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulonglong4 ulonglong4_a;
  ulonglong4 ulonglong4_a;
  // CHECK: cl::sycl::ulonglong4 ulonglong4_b = cl::sycl::ulonglong4(1, 2, 3, 4);
  ulonglong4 ulonglong4_b = make_ulonglong4(1, 2, 3, 4);
  // CHECK: cl::sycl::ulonglong4 ulonglong4_c = cl::sycl::ulonglong4(ulonglong4_b);
  ulonglong4 ulonglong4_c = ulonglong4(ulonglong4_b);
  // CHECK: cl::sycl::ulonglong4 ulonglong4_d(ulonglong4_c);
  ulonglong4 ulonglong4_d(ulonglong4_c);
  // CHECK: func3_ulonglong4(ulonglong4_b, cl::sycl::ulonglong4(ulonglong4_b), (cl::sycl::ulonglong4)ulonglong4_b);
  func3_ulonglong4(ulonglong4_b, ulonglong4(ulonglong4_b), (ulonglong4)ulonglong4_b);
  // CHECK: cl::sycl::ulonglong4 *ulonglong4_e;
  ulonglong4 *ulonglong4_e;
  // CHECK: cl::sycl::ulonglong4 *ulonglong4_f;
  ulonglong4 *ulonglong4_f;
  // CHECK: unsigned long long ulonglong4_g = static_cast<unsigned long long>(ulonglong4_c.x());
  unsigned long long ulonglong4_g = ulonglong4_c.x;
  // CHECK: ulonglong4_a.x() = static_cast<unsigned long long>(ulonglong4_d.x());
  ulonglong4_a.x = ulonglong4_d.x;
  // CHECK: if (static_cast<unsigned long long>(ulonglong4_b.x()) == static_cast<unsigned long long>(ulonglong4_d.x())) {}
  if (ulonglong4_b.x == ulonglong4_d.x) {}
  // CHECK: cl::sycl::ulonglong4 ulonglong4_h[16];
  ulonglong4 ulonglong4_h[16];
  // CHECK: cl::sycl::ulonglong4 ulonglong4_i[32];
  ulonglong4 ulonglong4_i[32];
  // CHECK: if (static_cast<unsigned long long>(ulonglong4_h[12].x()) == static_cast<unsigned long long>(ulonglong4_i[12].x())) {}
  if (ulonglong4_h[12].x == ulonglong4_i[12].x) {}
  // CHECK: ulonglong4_f = (cl::sycl::ulonglong4 *)ulonglong4_i;
  ulonglong4_f = (ulonglong4 *)ulonglong4_i;
  // CHECK: ulonglong4_a = (cl::sycl::ulonglong4)ulonglong4_c;
  ulonglong4_a = (ulonglong4)ulonglong4_c;
  // CHECK: ulonglong4_b = cl::sycl::ulonglong4(ulonglong4_b);
  ulonglong4_b = ulonglong4(ulonglong4_b);
  // CHECK: cl::sycl::ulonglong4 ulonglong4_j, ulonglong4_k, ulonglong4_l, ulonglong4_m[16], *ulonglong4_n[32];
  ulonglong4 ulonglong4_j, ulonglong4_k, ulonglong4_l, ulonglong4_m[16], *ulonglong4_n[32];
  // CHECK: int ulonglong4_o = sizeof(cl::sycl::ulonglong4);
  int ulonglong4_o = sizeof(ulonglong4);
  // CHECK: int unsigned long long_p = sizeof(unsigned long long);
  int unsigned long long_p = sizeof(unsigned long long);
  // CHECK: int ulonglong4_q = sizeof(ulonglong4_d);
  int ulonglong4_q = sizeof(ulonglong4_d);
  int *ulonglong4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulonglong4_e_buf_ct0 = dpct::get_buffer_and_offset(ulonglong4_e);
  // CHECK-NEXT:   size_t ulonglong4_e_offset_ct0 = ulonglong4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ulonglong4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::ulonglong4 *)ulonglong4_cast);
  // CHECK-NEXT:   size_t ulonglong4_cast_offset_ct1 = ulonglong4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ulonglong4_e_acc_ct0 = ulonglong4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ulonglong4_cast_acc_ct1 = ulonglong4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ulonglong4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::ulonglong4 *ulonglong4_e_ct0 = (cl::sycl::ulonglong4 *)(&ulonglong4_e_acc_ct0[0] + ulonglong4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::ulonglong4 *ulonglong4_cast_ct1 = (cl::sycl::ulonglong4 *)(&ulonglong4_cast_acc_ct1[0] + ulonglong4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ulonglong4(ulonglong4_e_ct0, ulonglong4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ulonglong4<<<1,1>>>(ulonglong4_e, (ulonglong4 *)ulonglong4_cast);
  return 0;
}

// CHECK: void func3_ushort1(unsigned short a, unsigned short b, unsigned short c) {
void func3_ushort1(ushort1 a, ushort1 b, ushort1 c) {
}
// CHECK: void func_ushort1(unsigned short a) {
void func_ushort1(ushort1 a) {
}
// CHECK: void kernel_ushort1(unsigned short *a, unsigned short *b) {
__global__ void kernel_ushort1(ushort1 *a, ushort1 *b) {
}

int main_ushort1() {
  // range default constructor does the right thing.
  // CHECK: unsigned short ushort1_a;
  ushort1 ushort1_a;
  // CHECK: unsigned short ushort1_b = unsigned short(1);
  ushort1 ushort1_b = make_ushort1(1);
  // CHECK: unsigned short ushort1_c = unsigned short(ushort1_b);
  ushort1 ushort1_c = ushort1(ushort1_b);
  // CHECK: unsigned short ushort1_d(ushort1_c);
  ushort1 ushort1_d(ushort1_c);
  // CHECK: func3_ushort1(ushort1_b, unsigned short(ushort1_b), (unsigned short)ushort1_b);
  func3_ushort1(ushort1_b, ushort1(ushort1_b), (ushort1)ushort1_b);
  // CHECK: unsigned short *ushort1_e;
  ushort1 *ushort1_e;
  // CHECK: unsigned short *ushort1_f;
  ushort1 *ushort1_f;
  // CHECK: unsigned short ushort1_g = static_cast<unsigned short>(ushort1_c);
  unsigned short ushort1_g = ushort1_c.x;
  // CHECK: ushort1_a = static_cast<unsigned short>(ushort1_d);
  ushort1_a.x = ushort1_d.x;
  // CHECK: if (static_cast<unsigned short>(ushort1_b) == static_cast<unsigned short>(ushort1_d)) {}
  if (ushort1_b.x == ushort1_d.x) {}
  // CHECK: unsigned short ushort1_h[16];
  ushort1 ushort1_h[16];
  // CHECK: unsigned short ushort1_i[32];
  ushort1 ushort1_i[32];
  // CHECK: if (static_cast<unsigned short>(ushort1_h[12]) == static_cast<unsigned short>(ushort1_i[12])) {}
  if (ushort1_h[12].x == ushort1_i[12].x) {}
  // CHECK: ushort1_f = (unsigned short *)ushort1_i;
  ushort1_f = (ushort1 *)ushort1_i;
  // CHECK: ushort1_a = (unsigned short)ushort1_c;
  ushort1_a = (ushort1)ushort1_c;
  // CHECK: ushort1_b = unsigned short(ushort1_b);
  ushort1_b = ushort1(ushort1_b);
  // CHECK: unsigned short ushort1_j, ushort1_k, ushort1_l, ushort1_m[16], *ushort1_n[32];
  ushort1 ushort1_j, ushort1_k, ushort1_l, ushort1_m[16], *ushort1_n[32];
  // CHECK: int ushort1_o = sizeof(unsigned short);
  int ushort1_o = sizeof(ushort1);
  // CHECK: int unsigned short_p = sizeof(unsigned short);
  int unsigned short_p = sizeof(unsigned short);
  // CHECK: int ushort1_q = sizeof(ushort1_d);
  int ushort1_q = sizeof(ushort1_d);
  int *ushort1_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ushort1_e_buf_ct0 = dpct::get_buffer_and_offset(ushort1_e);
  // CHECK-NEXT:   size_t ushort1_e_offset_ct0 = ushort1_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ushort1_cast_buf_ct1 = dpct::get_buffer_and_offset((unsigned short *)ushort1_cast);
  // CHECK-NEXT:   size_t ushort1_cast_offset_ct1 = ushort1_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ushort1_e_acc_ct0 = ushort1_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ushort1_cast_acc_ct1 = ushort1_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ushort1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           unsigned short *ushort1_e_ct0 = (unsigned short *)(&ushort1_e_acc_ct0[0] + ushort1_e_offset_ct0);
  // CHECK-NEXT:           unsigned short *ushort1_cast_ct1 = (unsigned short *)(&ushort1_cast_acc_ct1[0] + ushort1_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ushort1(ushort1_e_ct0, ushort1_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ushort1<<<1,1>>>(ushort1_e, (ushort1 *)ushort1_cast);
  return 0;
}

// CHECK: void func3_ushort2(cl::sycl::ushort2 a, cl::sycl::ushort2 b, cl::sycl::ushort2 c) {
void func3_ushort2(ushort2 a, ushort2 b, ushort2 c) {
}
// CHECK: void func_ushort2(cl::sycl::ushort2 a) {
void func_ushort2(ushort2 a) {
}
// CHECK: void kernel_ushort2(cl::sycl::ushort2 *a, cl::sycl::ushort2 *b) {
__global__ void kernel_ushort2(ushort2 *a, ushort2 *b) {
}

int main_ushort2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ushort2 ushort2_a;
  ushort2 ushort2_a;
  // CHECK: cl::sycl::ushort2 ushort2_b = cl::sycl::ushort2(1, 2);
  ushort2 ushort2_b = make_ushort2(1, 2);
  // CHECK: cl::sycl::ushort2 ushort2_c = cl::sycl::ushort2(ushort2_b);
  ushort2 ushort2_c = ushort2(ushort2_b);
  // CHECK: cl::sycl::ushort2 ushort2_d(ushort2_c);
  ushort2 ushort2_d(ushort2_c);
  // CHECK: func3_ushort2(ushort2_b, cl::sycl::ushort2(ushort2_b), (cl::sycl::ushort2)ushort2_b);
  func3_ushort2(ushort2_b, ushort2(ushort2_b), (ushort2)ushort2_b);
  // CHECK: cl::sycl::ushort2 *ushort2_e;
  ushort2 *ushort2_e;
  // CHECK: cl::sycl::ushort2 *ushort2_f;
  ushort2 *ushort2_f;
  // CHECK: unsigned short ushort2_g = static_cast<unsigned short>(ushort2_c.x());
  unsigned short ushort2_g = ushort2_c.x;
  // CHECK: ushort2_a.x() = static_cast<unsigned short>(ushort2_d.x());
  ushort2_a.x = ushort2_d.x;
  // CHECK: if (static_cast<unsigned short>(ushort2_b.x()) == static_cast<unsigned short>(ushort2_d.x())) {}
  if (ushort2_b.x == ushort2_d.x) {}
  // CHECK: cl::sycl::ushort2 ushort2_h[16];
  ushort2 ushort2_h[16];
  // CHECK: cl::sycl::ushort2 ushort2_i[32];
  ushort2 ushort2_i[32];
  // CHECK: if (static_cast<unsigned short>(ushort2_h[12].x()) == static_cast<unsigned short>(ushort2_i[12].x())) {}
  if (ushort2_h[12].x == ushort2_i[12].x) {}
  // CHECK: ushort2_f = (cl::sycl::ushort2 *)ushort2_i;
  ushort2_f = (ushort2 *)ushort2_i;
  // CHECK: ushort2_a = (cl::sycl::ushort2)ushort2_c;
  ushort2_a = (ushort2)ushort2_c;
  // CHECK: ushort2_b = cl::sycl::ushort2(ushort2_b);
  ushort2_b = ushort2(ushort2_b);
  // CHECK: cl::sycl::ushort2 ushort2_j, ushort2_k, ushort2_l, ushort2_m[16], *ushort2_n[32];
  ushort2 ushort2_j, ushort2_k, ushort2_l, ushort2_m[16], *ushort2_n[32];
  // CHECK: int ushort2_o = sizeof(cl::sycl::ushort2);
  int ushort2_o = sizeof(ushort2);
  // CHECK: int unsigned short_p = sizeof(unsigned short);
  int unsigned short_p = sizeof(unsigned short);
  // CHECK: int ushort2_q = sizeof(ushort2_d);
  int ushort2_q = sizeof(ushort2_d);
  int *ushort2_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ushort2_e_buf_ct0 = dpct::get_buffer_and_offset(ushort2_e);
  // CHECK-NEXT:   size_t ushort2_e_offset_ct0 = ushort2_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ushort2_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::ushort2 *)ushort2_cast);
  // CHECK-NEXT:   size_t ushort2_cast_offset_ct1 = ushort2_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ushort2_e_acc_ct0 = ushort2_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ushort2_cast_acc_ct1 = ushort2_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ushort2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::ushort2 *ushort2_e_ct0 = (cl::sycl::ushort2 *)(&ushort2_e_acc_ct0[0] + ushort2_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::ushort2 *ushort2_cast_ct1 = (cl::sycl::ushort2 *)(&ushort2_cast_acc_ct1[0] + ushort2_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ushort2(ushort2_e_ct0, ushort2_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ushort2<<<1,1>>>(ushort2_e, (ushort2 *)ushort2_cast);
  return 0;
}

// CHECK: void func3_ushort3(cl::sycl::ushort3 a, cl::sycl::ushort3 b, cl::sycl::ushort3 c) {
void func3_ushort3(ushort3 a, ushort3 b, ushort3 c) {
}
// CHECK: void func_ushort3(cl::sycl::ushort3 a) {
void func_ushort3(ushort3 a) {
}
// CHECK: void kernel_ushort3(cl::sycl::ushort3 *a, cl::sycl::ushort3 *b) {
__global__ void kernel_ushort3(ushort3 *a, ushort3 *b) {
}

int main_ushort3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ushort3 ushort3_a;
  ushort3 ushort3_a;
  // CHECK: cl::sycl::ushort3 ushort3_b = cl::sycl::ushort3(1, 2, 3);
  ushort3 ushort3_b = make_ushort3(1, 2, 3);
  // CHECK: cl::sycl::ushort3 ushort3_c = cl::sycl::ushort3(ushort3_b);
  ushort3 ushort3_c = ushort3(ushort3_b);
  // CHECK: cl::sycl::ushort3 ushort3_d(ushort3_c);
  ushort3 ushort3_d(ushort3_c);
  // CHECK: func3_ushort3(ushort3_b, cl::sycl::ushort3(ushort3_b), (cl::sycl::ushort3)ushort3_b);
  func3_ushort3(ushort3_b, ushort3(ushort3_b), (ushort3)ushort3_b);
  // CHECK: cl::sycl::ushort3 *ushort3_e;
  ushort3 *ushort3_e;
  // CHECK: cl::sycl::ushort3 *ushort3_f;
  ushort3 *ushort3_f;
  // CHECK: unsigned short ushort3_g = static_cast<unsigned short>(ushort3_c.x());
  unsigned short ushort3_g = ushort3_c.x;
  // CHECK: ushort3_a.x() = static_cast<unsigned short>(ushort3_d.x());
  ushort3_a.x = ushort3_d.x;
  // CHECK: if (static_cast<unsigned short>(ushort3_b.x()) == static_cast<unsigned short>(ushort3_d.x())) {}
  if (ushort3_b.x == ushort3_d.x) {}
  // CHECK: cl::sycl::ushort3 ushort3_h[16];
  ushort3 ushort3_h[16];
  // CHECK: cl::sycl::ushort3 ushort3_i[32];
  ushort3 ushort3_i[32];
  // CHECK: if (static_cast<unsigned short>(ushort3_h[12].x()) == static_cast<unsigned short>(ushort3_i[12].x())) {}
  if (ushort3_h[12].x == ushort3_i[12].x) {}
  // CHECK: ushort3_f = (cl::sycl::ushort3 *)ushort3_i;
  ushort3_f = (ushort3 *)ushort3_i;
  // CHECK: ushort3_a = (cl::sycl::ushort3)ushort3_c;
  ushort3_a = (ushort3)ushort3_c;
  // CHECK: ushort3_b = cl::sycl::ushort3(ushort3_b);
  ushort3_b = ushort3(ushort3_b);
  // CHECK: cl::sycl::ushort3 ushort3_j, ushort3_k, ushort3_l, ushort3_m[16], *ushort3_n[32];
  ushort3 ushort3_j, ushort3_k, ushort3_l, ushort3_m[16], *ushort3_n[32];
  // CHECK: int ushort3_o = sizeof(cl::sycl::ushort3);
  int ushort3_o = sizeof(ushort3);
  // CHECK: int unsigned short_p = sizeof(unsigned short);
  int unsigned short_p = sizeof(unsigned short);
  // CHECK: int ushort3_q = sizeof(ushort3_d);
  int ushort3_q = sizeof(ushort3_d);
  int *ushort3_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ushort3_e_buf_ct0 = dpct::get_buffer_and_offset(ushort3_e);
  // CHECK-NEXT:   size_t ushort3_e_offset_ct0 = ushort3_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ushort3_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::ushort3 *)ushort3_cast);
  // CHECK-NEXT:   size_t ushort3_cast_offset_ct1 = ushort3_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ushort3_e_acc_ct0 = ushort3_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ushort3_cast_acc_ct1 = ushort3_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ushort3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::ushort3 *ushort3_e_ct0 = (cl::sycl::ushort3 *)(&ushort3_e_acc_ct0[0] + ushort3_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::ushort3 *ushort3_cast_ct1 = (cl::sycl::ushort3 *)(&ushort3_cast_acc_ct1[0] + ushort3_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ushort3(ushort3_e_ct0, ushort3_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ushort3<<<1,1>>>(ushort3_e, (ushort3 *)ushort3_cast);
  return 0;
}

// CHECK: void func3_ushort4(cl::sycl::ushort4 a, cl::sycl::ushort4 b, cl::sycl::ushort4 c) {
void func3_ushort4(ushort4 a, ushort4 b, ushort4 c) {
}
// CHECK: void func_ushort4(cl::sycl::ushort4 a) {
void func_ushort4(ushort4 a) {
}
// CHECK: void kernel_ushort4(cl::sycl::ushort4 *a, cl::sycl::ushort4 *b) {
__global__ void kernel_ushort4(ushort4 *a, ushort4 *b) {
}

int main_ushort4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ushort4 ushort4_a;
  ushort4 ushort4_a;
  // CHECK: cl::sycl::ushort4 ushort4_b = cl::sycl::ushort4(1, 2, 3, 4);
  ushort4 ushort4_b = make_ushort4(1, 2, 3, 4);
  // CHECK: cl::sycl::ushort4 ushort4_c = cl::sycl::ushort4(ushort4_b);
  ushort4 ushort4_c = ushort4(ushort4_b);
  // CHECK: cl::sycl::ushort4 ushort4_d(ushort4_c);
  ushort4 ushort4_d(ushort4_c);
  // CHECK: func3_ushort4(ushort4_b, cl::sycl::ushort4(ushort4_b), (cl::sycl::ushort4)ushort4_b);
  func3_ushort4(ushort4_b, ushort4(ushort4_b), (ushort4)ushort4_b);
  // CHECK: cl::sycl::ushort4 *ushort4_e;
  ushort4 *ushort4_e;
  // CHECK: cl::sycl::ushort4 *ushort4_f;
  ushort4 *ushort4_f;
  // CHECK: unsigned short ushort4_g = static_cast<unsigned short>(ushort4_c.x());
  unsigned short ushort4_g = ushort4_c.x;
  // CHECK: ushort4_a.x() = static_cast<unsigned short>(ushort4_d.x());
  ushort4_a.x = ushort4_d.x;
  // CHECK: if (static_cast<unsigned short>(ushort4_b.x()) == static_cast<unsigned short>(ushort4_d.x())) {}
  if (ushort4_b.x == ushort4_d.x) {}
  // CHECK: cl::sycl::ushort4 ushort4_h[16];
  ushort4 ushort4_h[16];
  // CHECK: cl::sycl::ushort4 ushort4_i[32];
  ushort4 ushort4_i[32];
  // CHECK: if (static_cast<unsigned short>(ushort4_h[12].x()) == static_cast<unsigned short>(ushort4_i[12].x())) {}
  if (ushort4_h[12].x == ushort4_i[12].x) {}
  // CHECK: ushort4_f = (cl::sycl::ushort4 *)ushort4_i;
  ushort4_f = (ushort4 *)ushort4_i;
  // CHECK: ushort4_a = (cl::sycl::ushort4)ushort4_c;
  ushort4_a = (ushort4)ushort4_c;
  // CHECK: ushort4_b = cl::sycl::ushort4(ushort4_b);
  ushort4_b = ushort4(ushort4_b);
  // CHECK: cl::sycl::ushort4 ushort4_j, ushort4_k, ushort4_l, ushort4_m[16], *ushort4_n[32];
  ushort4 ushort4_j, ushort4_k, ushort4_l, ushort4_m[16], *ushort4_n[32];
  // CHECK: int ushort4_o = sizeof(cl::sycl::ushort4);
  int ushort4_o = sizeof(ushort4);
  // CHECK: int unsigned short_p = sizeof(unsigned short);
  int unsigned short_p = sizeof(unsigned short);
  // CHECK: int ushort4_q = sizeof(ushort4_d);
  int ushort4_q = sizeof(ushort4_d);
  int *ushort4_cast;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ushort4_e_buf_ct0 = dpct::get_buffer_and_offset(ushort4_e);
  // CHECK-NEXT:   size_t ushort4_e_offset_ct0 = ushort4_e_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> ushort4_cast_buf_ct1 = dpct::get_buffer_and_offset((cl::sycl::ushort4 *)ushort4_cast);
  // CHECK-NEXT:   size_t ushort4_cast_offset_ct1 = ushort4_cast_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto ushort4_e_acc_ct0 = ushort4_e_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto ushort4_cast_acc_ct1 = ushort4_cast_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto dpct_global_range = cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       auto dpct_local_range = cl::sycl::range<3>(1, 1, 1);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class kernel_ushort4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           cl::sycl::ushort4 *ushort4_e_ct0 = (cl::sycl::ushort4 *)(&ushort4_e_acc_ct0[0] + ushort4_e_offset_ct0);
  // CHECK-NEXT:           cl::sycl::ushort4 *ushort4_cast_ct1 = (cl::sycl::ushort4 *)(&ushort4_cast_acc_ct1[0] + ushort4_cast_offset_ct1);
  // CHECK-NEXT:           kernel_ushort4(ushort4_e_ct0, ushort4_cast_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK: }
  kernel_ushort4<<<1,1>>>(ushort4_e, (ushort4 *)ushort4_cast);
  return 0;
}
