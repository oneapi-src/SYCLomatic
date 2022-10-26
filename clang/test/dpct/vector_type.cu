// RUN: dpct --format-range=none --usm-level=none -out-root %T/vector_type %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/vector_type/vector_type.dp.cpp --match-full-lines %s

#include <vector>

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
  // CHECK: signed char char1_g = char1_c;
  signed char char1_g = char1_c.x;
  // CHECK: char1_a = char1_d;
  char1_a.x = char1_d.x;
  // CHECK: if (char1_b == char1_d) {}
  if (char1_b.x == char1_d.x) {}
  // CHECK: char char1_h[16];
  char1 char1_h[16];
  // CHECK: char char1_i[32];
  char1 char1_i[32];
  // CHECK: if (char1_h[12] == char1_i[12]) {}
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
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<char *> char1_e_acc_ct0(char1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<char *> char1_cast_acc_ct1((char *)char1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_char1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_char1(char1_e_acc_ct0.get_raw_pointer(), char1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_char1<<<1,1>>>(char1_e, (char1 *)char1_cast);
  // CHECK: char char1_r = (char){1};
  // CHECK-NEXT: auto char1_s = (char){1};
  char1 char1_r = (char1){1};
  auto char1_s = (char1){1};
  return 0;
}

// CHECK: void func3_char2(sycl::char2 a, sycl::char2 b, sycl::char2 c) {
void func3_char2(char2 a, char2 b, char2 c) {
}
// CHECK: void func_char2(sycl::char2 a) {
void func_char2(char2 a) {
}
// CHECK: void kernel_char2(sycl::char2 *a, sycl::char2 *b) {
__global__ void kernel_char2(char2 *a, char2 *b) {
}

int main_char2() {
  // range default constructor does the right thing.
  // CHECK: sycl::char2 char2_a;
  char2 char2_a;
  // CHECK: sycl::char2 char2_b = sycl::char2(1, 2);
  char2 char2_b = make_char2(1, 2);
  // CHECK: sycl::char2 char2_c = sycl::char2(char2_b);
  char2 char2_c = char2(char2_b);
  // CHECK: sycl::char2 char2_d(char2_c);
  char2 char2_d(char2_c);
  // CHECK: func3_char2(char2_b, sycl::char2(char2_b), (sycl::char2)char2_b);
  func3_char2(char2_b, char2(char2_b), (char2)char2_b);
  // CHECK: sycl::char2 *char2_e;
  char2 *char2_e;
  // CHECK: sycl::char2 *char2_f;
  char2 *char2_f;
  // CHECK: signed char char2_g = char2_c.x();
  signed char char2_g = char2_c.x;
  // CHECK: char2_a.x() = char2_d.x();
  char2_a.x = char2_d.x;
  // CHECK: if (char2_b.x() == char2_d.x()) {}
  if (char2_b.x == char2_d.x) {}
  // CHECK: sycl::char2 char2_h[16];
  char2 char2_h[16];
  // CHECK: sycl::char2 char2_i[32];
  char2 char2_i[32];
  // CHECK: if (char2_h[12].x() == char2_i[12].x()) {}
  if (char2_h[12].x == char2_i[12].x) {}
  // CHECK: char2_f = (sycl::char2 *)char2_i;
  char2_f = (char2 *)char2_i;
  // CHECK: char2_a = (sycl::char2)char2_c;
  char2_a = (char2)char2_c;
  // CHECK: char2_b = sycl::char2(char2_b);
  char2_b = char2(char2_b);
  // CHECK: sycl::char2 char2_j, char2_k, char2_l, char2_m[16], *char2_n[32];
  char2 char2_j, char2_k, char2_l, char2_m[16], *char2_n[32];
  // CHECK: int char2_o = sizeof(sycl::char2);
  int char2_o = sizeof(char2);
  // CHECK: int signed char_p = sizeof(signed char);
  int signed char_p = sizeof(signed char);
  // CHECK: int char2_q = sizeof(char2_d);
  int char2_q = sizeof(char2_d);
  int *char2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::char2 *> char2_e_acc_ct0(char2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::char2 *> char2_cast_acc_ct1((sycl::char2 *)char2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_char2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_char2(char2_e_acc_ct0.get_raw_pointer(), char2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_char2<<<1,1>>>(char2_e, (char2 *)char2_cast);
  // CHECK: sycl::char2 char2_r = (sycl::char2){1,1};
  // CHECK-NEXT: auto char2_s = (sycl::char2){1,1};
  char2 char2_r = (char2){1,1};
  auto char2_s = (char2){1,1};
  return 0;
}

// CHECK: void func3_char3(sycl::char3 a, sycl::char3 b, sycl::char3 c) {
void func3_char3(char3 a, char3 b, char3 c) {
}
// CHECK: void func_char3(sycl::char3 a) {
void func_char3(char3 a) {
}
// CHECK: void kernel_char3(sycl::char3 *a, sycl::char3 *b) {
__global__ void kernel_char3(char3 *a, char3 *b) {
}

int main_char3() {
  // range default constructor does the right thing.
  // CHECK: sycl::char3 char3_a;
  char3 char3_a;
  // CHECK: sycl::char3 char3_b = sycl::char3(1, 2, 3);
  char3 char3_b = make_char3(1, 2, 3);
  // CHECK: sycl::char3 char3_c = sycl::char3(char3_b);
  char3 char3_c = char3(char3_b);
  // CHECK: sycl::char3 char3_d(char3_c);
  char3 char3_d(char3_c);
  // CHECK: func3_char3(char3_b, sycl::char3(char3_b), (sycl::char3)char3_b);
  func3_char3(char3_b, char3(char3_b), (char3)char3_b);
  // CHECK: sycl::char3 *char3_e;
  char3 *char3_e;
  // CHECK: sycl::char3 *char3_f;
  char3 *char3_f;
  // CHECK: signed char char3_g = char3_c.x();
  signed char char3_g = char3_c.x;
  // CHECK: char3_a.x() = char3_d.x();
  char3_a.x = char3_d.x;
  // CHECK: if (char3_b.x() == char3_d.x()) {}
  if (char3_b.x == char3_d.x) {}
  // CHECK: sycl::char3 char3_h[16];
  char3 char3_h[16];
  // CHECK: sycl::char3 char3_i[32];
  char3 char3_i[32];
  // CHECK: if (char3_h[12].x() == char3_i[12].x()) {}
  if (char3_h[12].x == char3_i[12].x) {}
  // CHECK: char3_f = (sycl::char3 *)char3_i;
  char3_f = (char3 *)char3_i;
  // CHECK: char3_a = (sycl::char3)char3_c;
  char3_a = (char3)char3_c;
  // CHECK: char3_b = sycl::char3(char3_b);
  char3_b = char3(char3_b);
  // CHECK: sycl::char3 char3_j, char3_k, char3_l, char3_m[16], *char3_n[32];
  char3 char3_j, char3_k, char3_l, char3_m[16], *char3_n[32];
  // CHECK: int char3_o = sizeof(sycl::char3);
  int char3_o = sizeof(char3);
  // CHECK: int signed char_p = sizeof(signed char);
  int signed char_p = sizeof(signed char);
  // CHECK: int char3_q = sizeof(char3_d);
  int char3_q = sizeof(char3_d);
  int *char3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::char3 *> char3_e_acc_ct0(char3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::char3 *> char3_cast_acc_ct1((sycl::char3 *)char3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_char3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_char3(char3_e_acc_ct0.get_raw_pointer(), char3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_char3<<<1,1>>>(char3_e, (char3 *)char3_cast);
  // CHECK: sycl::char3 char3_r = (sycl::char3){1,1,1};
  // CHECK-NEXT: auto char3_s = (sycl::char3){1,1,1};
  char3 char3_r = (char3){1,1,1};
  auto char3_s = (char3){1,1,1};
  return 0;
}

// CHECK: void func3_char4(sycl::char4 a, sycl::char4 b, sycl::char4 c) {
void func3_char4(char4 a, char4 b, char4 c) {
}
// CHECK: void func_char4(sycl::char4 a) {
void func_char4(char4 a) {
}
// CHECK: void kernel_char4(sycl::char4 *a, sycl::char4 *b) {
__global__ void kernel_char4(char4 *a, char4 *b) {
}

int main_char4() {
  // range default constructor does the right thing.
  // CHECK: sycl::char4 char4_a;
  char4 char4_a;
  // CHECK: sycl::char4 char4_b = sycl::char4(1, 2, 3, 4);
  char4 char4_b = make_char4(1, 2, 3, 4);
  // CHECK: sycl::char4 char4_c = sycl::char4(char4_b);
  char4 char4_c = char4(char4_b);
  // CHECK: sycl::char4 char4_d(char4_c);
  char4 char4_d(char4_c);
  // CHECK: func3_char4(char4_b, sycl::char4(char4_b), (sycl::char4)char4_b);
  func3_char4(char4_b, char4(char4_b), (char4)char4_b);
  // CHECK: sycl::char4 *char4_e;
  char4 *char4_e;
  // CHECK: sycl::char4 *char4_f;
  char4 *char4_f;
  // CHECK: signed char char4_g = char4_c.x();
  signed char char4_g = char4_c.x;
  // CHECK: char4_a.x() = char4_d.x();
  char4_a.x = char4_d.x;
  // CHECK: if (char4_b.x() == char4_d.x()) {}
  if (char4_b.x == char4_d.x) {}
  // CHECK: sycl::char4 char4_h[16];
  char4 char4_h[16];
  // CHECK: sycl::char4 char4_i[32];
  char4 char4_i[32];
  // CHECK: if (char4_h[12].x() == char4_i[12].x()) {}
  if (char4_h[12].x == char4_i[12].x) {}
  // CHECK: char4_f = (sycl::char4 *)char4_i;
  char4_f = (char4 *)char4_i;
  // CHECK: char4_a = (sycl::char4)char4_c;
  char4_a = (char4)char4_c;
  // CHECK: char4_b = sycl::char4(char4_b);
  char4_b = char4(char4_b);
  // CHECK: sycl::char4 char4_j, char4_k, char4_l, char4_m[16], *char4_n[32];
  char4 char4_j, char4_k, char4_l, char4_m[16], *char4_n[32];
  // CHECK: int char4_o = sizeof(sycl::char4);
  int char4_o = sizeof(char4);
  // CHECK: int signed char_p = sizeof(signed char);
  int signed char_p = sizeof(signed char);
  // CHECK: int char4_q = sizeof(char4_d);
  int char4_q = sizeof(char4_d);
  int *char4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::char4 *> char4_e_acc_ct0(char4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::char4 *> char4_cast_acc_ct1((sycl::char4 *)char4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_char4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_char4(char4_e_acc_ct0.get_raw_pointer(), char4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_char4<<<1,1>>>(char4_e, (char4 *)char4_cast);
  // CHECK: sycl::char4 char4_r = (sycl::char4){1,1,1,1};
  // CHECK-NEXT: auto char4_s = (sycl::char4){1,1,1,1};
  char4 char4_r = (char4){1,1,1,1};
  auto char4_s = (char4){1,1,1,1};
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
  // CHECK: double double1_g = double1_c;
  double double1_g = double1_c.x;
  // CHECK: double1_a = double1_d;
  double1_a.x = double1_d.x;
  // CHECK: if (double1_b == double1_d) {}
  if (double1_b.x == double1_d.x) {}
  // CHECK: double double1_h[16];
  double1 double1_h[16];
  // CHECK: double double1_i[32];
  double1 double1_i[32];
  // CHECK: if (double1_h[12] == double1_i[12]) {}
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
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<double *> double1_e_acc_ct0(double1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<double *> double1_cast_acc_ct1((double *)double1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_double1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_double1(double1_e_acc_ct0.get_raw_pointer(), double1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_double1<<<1,1>>>(double1_e, (double1 *)double1_cast);
  // CHECK: double double1_r = (double){1};
  // CHECK-NEXT: auto double1_s = (double){1};
  double1 double1_r = (double1){1};
  auto double1_s = (double1){1};
  return 0;
}

// CHECK: void func3_double2(sycl::double2 a, sycl::double2 b, sycl::double2 c) {
void func3_double2(double2 a, double2 b, double2 c) {
}
// CHECK: void func_double2(sycl::double2 a) {
void func_double2(double2 a) {
}
// CHECK: void kernel_double2(sycl::double2 *a, sycl::double2 *b) {
__global__ void kernel_double2(double2 *a, double2 *b) {
}

int main_double2() {
  // range default constructor does the right thing.
  // CHECK: sycl::double2 double2_a;
  double2 double2_a;
  // CHECK: sycl::double2 double2_b = sycl::double2(1, 2);
  double2 double2_b = make_double2(1, 2);
  // CHECK: sycl::double2 double2_c = sycl::double2(double2_b);
  double2 double2_c = double2(double2_b);
  // CHECK: sycl::double2 double2_d(double2_c);
  double2 double2_d(double2_c);
  // CHECK: func3_double2(double2_b, sycl::double2(double2_b), (sycl::double2)double2_b);
  func3_double2(double2_b, double2(double2_b), (double2)double2_b);
  // CHECK: sycl::double2 *double2_e;
  double2 *double2_e;
  // CHECK: sycl::double2 *double2_f;
  double2 *double2_f;
  // CHECK: double double2_g = double2_c.x();
  double double2_g = double2_c.x;
  // CHECK: double2_a.x() = double2_d.x();
  double2_a.x = double2_d.x;
  // CHECK: if (double2_b.x() == double2_d.x()) {}
  if (double2_b.x == double2_d.x) {}
  // CHECK: sycl::double2 double2_h[16];
  double2 double2_h[16];
  // CHECK: sycl::double2 double2_i[32];
  double2 double2_i[32];
  // CHECK: if (double2_h[12].x() == double2_i[12].x()) {}
  if (double2_h[12].x == double2_i[12].x) {}
  // CHECK: double2_f = (sycl::double2 *)double2_i;
  double2_f = (double2 *)double2_i;
  // CHECK: double2_a = (sycl::double2)double2_c;
  double2_a = (double2)double2_c;
  // CHECK: double2_b = sycl::double2(double2_b);
  double2_b = double2(double2_b);
  // CHECK: sycl::double2 double2_j, double2_k, double2_l, double2_m[16], *double2_n[32];
  double2 double2_j, double2_k, double2_l, double2_m[16], *double2_n[32];
  // CHECK: int double2_o = sizeof(sycl::double2);
  int double2_o = sizeof(double2);
  // CHECK: int double_p = sizeof(double);
  int double_p = sizeof(double);
  // CHECK: int double2_q = sizeof(double2_d);
  int double2_q = sizeof(double2_d);
  int *double2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::double2 *> double2_e_acc_ct0(double2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::double2 *> double2_cast_acc_ct1((sycl::double2 *)double2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_double2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_double2(double2_e_acc_ct0.get_raw_pointer(), double2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_double2<<<1,1>>>(double2_e, (double2 *)double2_cast);
  // CHECK: sycl::double2 double2_r = (sycl::double2){1,1};
  // CHECK-NEXT: auto double2_s = (sycl::double2){1,1};
  double2 double2_r = (double2){1,1};
  auto double2_s = (double2){1,1};
  return 0;
}

// CHECK: void func3_double3(sycl::double3 a, sycl::double3 b, sycl::double3 c) {
void func3_double3(double3 a, double3 b, double3 c) {
}
// CHECK: void func_double3(sycl::double3 a) {
void func_double3(double3 a) {
}
// CHECK: void kernel_double3(sycl::double3 *a, sycl::double3 *b) {
__global__ void kernel_double3(double3 *a, double3 *b) {
}

int main_double3() {
  // range default constructor does the right thing.
  // CHECK: sycl::double3 double3_a;
  double3 double3_a;
  // CHECK: sycl::double3 double3_b = sycl::double3(1, 2, 3);
  double3 double3_b = make_double3(1, 2, 3);
  // CHECK: sycl::double3 double3_c = sycl::double3(double3_b);
  double3 double3_c = double3(double3_b);
  // CHECK: sycl::double3 double3_d(double3_c);
  double3 double3_d(double3_c);
  // CHECK: func3_double3(double3_b, sycl::double3(double3_b), (sycl::double3)double3_b);
  func3_double3(double3_b, double3(double3_b), (double3)double3_b);
  // CHECK: sycl::double3 *double3_e;
  double3 *double3_e;
  // CHECK: sycl::double3 *double3_f;
  double3 *double3_f;
  // CHECK: double double3_g = double3_c.x();
  double double3_g = double3_c.x;
  // CHECK: double3_a.x() = double3_d.x();
  double3_a.x = double3_d.x;
  // CHECK: if (double3_b.x() == double3_d.x()) {}
  if (double3_b.x == double3_d.x) {}
  // CHECK: sycl::double3 double3_h[16];
  double3 double3_h[16];
  // CHECK: sycl::double3 double3_i[32];
  double3 double3_i[32];
  // CHECK: if (double3_h[12].x() == double3_i[12].x()) {}
  if (double3_h[12].x == double3_i[12].x) {}
  // CHECK: double3_f = (sycl::double3 *)double3_i;
  double3_f = (double3 *)double3_i;
  // CHECK: double3_a = (sycl::double3)double3_c;
  double3_a = (double3)double3_c;
  // CHECK: double3_b = sycl::double3(double3_b);
  double3_b = double3(double3_b);
  // CHECK: sycl::double3 double3_j, double3_k, double3_l, double3_m[16], *double3_n[32];
  double3 double3_j, double3_k, double3_l, double3_m[16], *double3_n[32];
  // CHECK: int double3_o = sizeof(sycl::double3);
  int double3_o = sizeof(double3);
  // CHECK: int double_p = sizeof(double);
  int double_p = sizeof(double);
  // CHECK: int double3_q = sizeof(double3_d);
  int double3_q = sizeof(double3_d);
  int *double3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::double3 *> double3_e_acc_ct0(double3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::double3 *> double3_cast_acc_ct1((sycl::double3 *)double3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_double3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_double3(double3_e_acc_ct0.get_raw_pointer(), double3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_double3<<<1,1>>>(double3_e, (double3 *)double3_cast);
  // CHECK: sycl::double3 double3_r = (sycl::double3){1,1,1};
  // CHECK-NEXT: auto double3_s = (sycl::double3){1,1,1};
  double3 double3_r = (double3){1,1,1};
  auto double3_s = (double3){1,1,1};
  return 0;
}

// CHECK: void func3_double4(sycl::double4 a, sycl::double4 b, sycl::double4 c) {
void func3_double4(double4 a, double4 b, double4 c) {
}
// CHECK: void func_double4(sycl::double4 a) {
void func_double4(double4 a) {
}
// CHECK: void kernel_double4(sycl::double4 *a, sycl::double4 *b) {
__global__ void kernel_double4(double4 *a, double4 *b) {
}

int main_double4() {
  // range default constructor does the right thing.
  // CHECK: sycl::double4 double4_a;
  double4 double4_a;
  // CHECK: sycl::double4 double4_b = sycl::double4(1, 2, 3, 4);
  double4 double4_b = make_double4(1, 2, 3, 4);
  // CHECK: sycl::double4 double4_c = sycl::double4(double4_b);
  double4 double4_c = double4(double4_b);
  // CHECK: sycl::double4 double4_d(double4_c);
  double4 double4_d(double4_c);
  // CHECK: func3_double4(double4_b, sycl::double4(double4_b), (sycl::double4)double4_b);
  func3_double4(double4_b, double4(double4_b), (double4)double4_b);
  // CHECK: sycl::double4 *double4_e;
  double4 *double4_e;
  // CHECK: sycl::double4 *double4_f;
  double4 *double4_f;
  // CHECK: double double4_g = double4_c.x();
  double double4_g = double4_c.x;
  // CHECK: double4_a.x() = double4_d.x();
  double4_a.x = double4_d.x;
  // CHECK: if (double4_b.x() == double4_d.x()) {}
  if (double4_b.x == double4_d.x) {}
  // CHECK: sycl::double4 double4_h[16];
  double4 double4_h[16];
  // CHECK: sycl::double4 double4_i[32];
  double4 double4_i[32];
  // CHECK: if (double4_h[12].x() == double4_i[12].x()) {}
  if (double4_h[12].x == double4_i[12].x) {}
  // CHECK: double4_f = (sycl::double4 *)double4_i;
  double4_f = (double4 *)double4_i;
  // CHECK: double4_a = (sycl::double4)double4_c;
  double4_a = (double4)double4_c;
  // CHECK: double4_b = sycl::double4(double4_b);
  double4_b = double4(double4_b);
  // CHECK: sycl::double4 double4_j, double4_k, double4_l, double4_m[16], *double4_n[32];
  double4 double4_j, double4_k, double4_l, double4_m[16], *double4_n[32];
  // CHECK: int double4_o = sizeof(sycl::double4);
  int double4_o = sizeof(double4);
  // CHECK: int double_p = sizeof(double);
  int double_p = sizeof(double);
  // CHECK: int double4_q = sizeof(double4_d);
  int double4_q = sizeof(double4_d);
  int *double4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::double4 *> double4_e_acc_ct0(double4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::double4 *> double4_cast_acc_ct1((sycl::double4 *)double4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_double4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_double4(double4_e_acc_ct0.get_raw_pointer(), double4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_double4<<<1,1>>>(double4_e, (double4 *)double4_cast);
  // CHECK: sycl::double4 double4_r = (sycl::double4){1,1,1,1};
  // CHECK-NEXT: auto double4_s = (sycl::double4){1,1,1,1};
  double4 double4_r = (double4){1,1,1,1};
  auto double4_s = (double4){1,1,1,1};
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
  // CHECK: float float1_g = float1_c;
  float float1_g = float1_c.x;
  // CHECK: float1_a = float1_d;
  float1_a.x = float1_d.x;
  // CHECK: if (float1_b == float1_d) {}
  if (float1_b.x == float1_d.x) {}
  // CHECK: float float1_h[16];
  float1 float1_h[16];
  // CHECK: float float1_i[32];
  float1 float1_i[32];
  // CHECK: if (float1_h[12] == float1_i[12]) {}
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
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<float *> float1_e_acc_ct0(float1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<float *> float1_cast_acc_ct1((float *)float1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_float1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_float1(float1_e_acc_ct0.get_raw_pointer(), float1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_float1<<<1,1>>>(float1_e, (float1 *)float1_cast);
  // CHECK: float float1_r = (float){1};
  // CHECK-NEXT: auto float1_s = (float){1};
  float1 float1_r = (float1){1};
  auto float1_s = (float1){1};
  return 0;
}

// CHECK: void func3_float2(sycl::float2 a, sycl::float2 b, sycl::float2 c) {
void func3_float2(float2 a, float2 b, float2 c) {
}
// CHECK: void func_float2(sycl::float2 a) {
void func_float2(float2 a) {
}
// CHECK: void kernel_float2(sycl::float2 *a, sycl::float2 *b) {
__global__ void kernel_float2(float2 *a, float2 *b) {
}

int main_float2() {
  // range default constructor does the right thing.
  // CHECK: sycl::float2 float2_a;
  float2 float2_a;
  // CHECK: sycl::float2 float2_b = sycl::float2(1, 2);
  float2 float2_b = make_float2(1, 2);
  // CHECK: sycl::float2 float2_c = sycl::float2(float2_b);
  float2 float2_c = float2(float2_b);
  // CHECK: sycl::float2 float2_d(float2_c);
  float2 float2_d(float2_c);
  // CHECK: func3_float2(float2_b, sycl::float2(float2_b), (sycl::float2)float2_b);
  func3_float2(float2_b, float2(float2_b), (float2)float2_b);
  // CHECK: sycl::float2 *float2_e;
  float2 *float2_e;
  // CHECK: sycl::float2 *float2_f;
  float2 *float2_f;
  // CHECK: float float2_g = float2_c.x();
  float float2_g = float2_c.x;
  // CHECK: float2_a.x() = float2_d.x();
  float2_a.x = float2_d.x;
  // CHECK: if (float2_b.x() == float2_d.x()) {}
  if (float2_b.x == float2_d.x) {}
  // CHECK: sycl::float2 float2_h[16];
  float2 float2_h[16];
  // CHECK: sycl::float2 float2_i[32];
  float2 float2_i[32];
  // CHECK: if (float2_h[12].x() == float2_i[12].x()) {}
  if (float2_h[12].x == float2_i[12].x) {}
  // CHECK: float2_f = (sycl::float2 *)float2_i;
  float2_f = (float2 *)float2_i;
  // CHECK: float2_a = (sycl::float2)float2_c;
  float2_a = (float2)float2_c;
  // CHECK: float2_b = sycl::float2(float2_b);
  float2_b = float2(float2_b);
  // CHECK: sycl::float2 float2_j, float2_k, float2_l, float2_m[16], *float2_n[32];
  float2 float2_j, float2_k, float2_l, float2_m[16], *float2_n[32];
  // CHECK: int float2_o = sizeof(sycl::float2);
  int float2_o = sizeof(float2);
  // CHECK: int float_p = sizeof(float);
  int float_p = sizeof(float);
  // CHECK: int float2_q = sizeof(float2_d);
  int float2_q = sizeof(float2_d);
  int *float2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::float2 *> float2_e_acc_ct0(float2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::float2 *> float2_cast_acc_ct1((sycl::float2 *)float2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_float2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_float2(float2_e_acc_ct0.get_raw_pointer(), float2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_float2<<<1,1>>>(float2_e, (float2 *)float2_cast);
  // CHECK: sycl::float2 float2_r = (sycl::float2){1,1};
  // CHECK-NEXT: auto float2_s = (sycl::float2){1,1};
  float2 float2_r = (float2){1,1};
  auto float2_s = (float2){1,1};
  return 0;
}

// CHECK: void func3_float3(sycl::float3 a, sycl::float3 b, sycl::float3 c) {
void func3_float3(float3 a, float3 b, float3 c) {
}
// CHECK: void func_float3(sycl::float3 a) {
void func_float3(float3 a) {
}
// CHECK: void kernel_float3(sycl::float3 *a, sycl::float3 *b) {
__global__ void kernel_float3(float3 *a, float3 *b) {
}

int main_float3() {
  // range default constructor does the right thing.
  // CHECK: sycl::float3 float3_a;
  float3 float3_a;
  // CHECK: sycl::float3 float3_b = sycl::float3(1, 2, 3);
  float3 float3_b = make_float3(1, 2, 3);
  // CHECK: sycl::float3 float3_c = sycl::float3(float3_b);
  float3 float3_c = float3(float3_b);
  // CHECK: sycl::float3 float3_d(float3_c);
  float3 float3_d(float3_c);
  // CHECK: func3_float3(float3_b, sycl::float3(float3_b), (sycl::float3)float3_b);
  func3_float3(float3_b, float3(float3_b), (float3)float3_b);
  // CHECK: sycl::float3 *float3_e;
  float3 *float3_e;
  // CHECK: sycl::float3 *float3_f;
  float3 *float3_f;
  // CHECK: float float3_g = float3_c.x();
  float float3_g = float3_c.x;
  // CHECK: float3_a.x() = float3_d.x();
  float3_a.x = float3_d.x;
  // CHECK: if (float3_b.x() == float3_d.x()) {}
  if (float3_b.x == float3_d.x) {}
  // CHECK: sycl::float3 float3_h[16];
  float3 float3_h[16];
  // CHECK: sycl::float3 float3_i[32];
  float3 float3_i[32];
  // CHECK: if (float3_h[12].x() == float3_i[12].x()) {}
  if (float3_h[12].x == float3_i[12].x) {}
  // CHECK: float3_f = (sycl::float3 *)float3_i;
  float3_f = (float3 *)float3_i;
  // CHECK: float3_a = (sycl::float3)float3_c;
  float3_a = (float3)float3_c;
  // CHECK: float3_b = sycl::float3(float3_b);
  float3_b = float3(float3_b);
  // CHECK: sycl::float3 float3_j, float3_k, float3_l, float3_m[16], *float3_n[32];
  float3 float3_j, float3_k, float3_l, float3_m[16], *float3_n[32];
  // CHECK: int float3_o = sizeof(sycl::float3);
  int float3_o = sizeof(float3);
  // CHECK: int float_p = sizeof(float);
  int float_p = sizeof(float);
  // CHECK: int float3_q = sizeof(float3_d);
  int float3_q = sizeof(float3_d);
  int *float3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::float3 *> float3_e_acc_ct0(float3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::float3 *> float3_cast_acc_ct1((sycl::float3 *)float3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_float3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_float3(float3_e_acc_ct0.get_raw_pointer(), float3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_float3<<<1,1>>>(float3_e, (float3 *)float3_cast);
  // CHECK: sycl::float3 float3_r = (sycl::float3){1,1,1};
  // CHECK-NEXT: auto float3_s = (sycl::float3){1,1,1};
  float3 float3_r = (float3){1,1,1};
  auto float3_s = (float3){1,1,1};
  return 0;
}

// CHECK: void func3_float4(sycl::float4 a, sycl::float4 b, sycl::float4 c) {
void func3_float4(float4 a, float4 b, float4 c) {
}
// CHECK: void func_float4(sycl::float4 a) {
void func_float4(float4 a) {
}
// CHECK: void kernel_float4(sycl::float4 *a, sycl::float4 *b) {
__global__ void kernel_float4(float4 *a, float4 *b) {
}

int main_float4() {
  // range default constructor does the right thing.
  // CHECK: sycl::float4 float4_a;
  float4 float4_a;
  // CHECK: sycl::float4 float4_b = sycl::float4(1, 2, 3, 4);
  float4 float4_b = make_float4(1, 2, 3, 4);
  // CHECK: sycl::float4 float4_c = sycl::float4(float4_b);
  float4 float4_c = float4(float4_b);
  // CHECK: sycl::float4 float4_d(float4_c);
  float4 float4_d(float4_c);
  // CHECK: func3_float4(float4_b, sycl::float4(float4_b), (sycl::float4)float4_b);
  func3_float4(float4_b, float4(float4_b), (float4)float4_b);
  // CHECK: sycl::float4 *float4_e;
  float4 *float4_e;
  // CHECK: sycl::float4 *float4_f;
  float4 *float4_f;
  // CHECK: float float4_g = float4_c.x();
  float float4_g = float4_c.x;
  // CHECK: float4_a.x() = float4_d.x();
  float4_a.x = float4_d.x;
  // CHECK: if (float4_b.x() == float4_d.x()) {}
  if (float4_b.x == float4_d.x) {}
  // CHECK: sycl::float4 float4_h[16];
  float4 float4_h[16];
  // CHECK: sycl::float4 float4_i[32];
  float4 float4_i[32];
  // CHECK: if (float4_h[12].x() == float4_i[12].x()) {}
  if (float4_h[12].x == float4_i[12].x) {}
  // CHECK: float4_f = (sycl::float4 *)float4_i;
  float4_f = (float4 *)float4_i;
  // CHECK: float4_a = (sycl::float4)float4_c;
  float4_a = (float4)float4_c;
  // CHECK: float4_b = sycl::float4(float4_b);
  float4_b = float4(float4_b);
  // CHECK: sycl::float4 float4_j, float4_k, float4_l, float4_m[16], *float4_n[32];
  float4 float4_j, float4_k, float4_l, float4_m[16], *float4_n[32];
  // CHECK: int float4_o = sizeof(sycl::float4);
  int float4_o = sizeof(float4);
  // CHECK: int float_p = sizeof(float);
  int float_p = sizeof(float);
  // CHECK: int float4_q = sizeof(float4_d);
  int float4_q = sizeof(float4_d);
  int *float4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::float4 *> float4_e_acc_ct0(float4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::float4 *> float4_cast_acc_ct1((sycl::float4 *)float4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_float4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_float4(float4_e_acc_ct0.get_raw_pointer(), float4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_float4<<<1,1>>>(float4_e, (float4 *)float4_cast);
  // CHECK: sycl::float4 float4_r = (sycl::float4){1,1,1,1};
  // CHECK-NEXT: auto float4_s = (sycl::float4){1,1,1,1};
  float4 float4_r = (float4){1,1,1,1};
  auto float4_s = (float4){1,1,1,1};
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
  // CHECK: int int1_g = int1_c;
  int int1_g = int1_c.x;
  // CHECK: int1_a = int1_d;
  int1_a.x = int1_d.x;
  // CHECK: if (int1_b == int1_d) {}
  if (int1_b.x == int1_d.x) {}
  // CHECK: int int1_h[16];
  int1 int1_h[16];
  // CHECK: int int1_i[32];
  int1 int1_i[32];
  // CHECK: if (int1_h[12] == int1_i[12]) {}
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
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<int *> int1_e_acc_ct0(int1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<int *> int1_cast_acc_ct1((int *)int1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_int1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_int1(int1_e_acc_ct0.get_raw_pointer(), int1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_int1<<<1,1>>>(int1_e, (int1 *)int1_cast);
  // CHECK: int int1_r = (int){1};
  // CHECK-NEXT: auto int1_s = (int){1};
  int1 int1_r = (int1){1};
  auto int1_s = (int1){1};
  return 0;
}

// CHECK: void func3_int2(sycl::int2 a, sycl::int2 b, sycl::int2 c) {
void func3_int2(int2 a, int2 b, int2 c) {
}
// CHECK: void func_int2(sycl::int2 a) {
void func_int2(int2 a) {
}
// CHECK: void kernel_int2(sycl::int2 *a, sycl::int2 *b) {
__global__ void kernel_int2(int2 *a, int2 *b) {
}

int main_int2() {
  // range default constructor does the right thing.
  // CHECK: sycl::int2 int2_a;
  int2 int2_a;
  // CHECK: sycl::int2 int2_b = sycl::int2(1, 2);
  int2 int2_b = make_int2(1, 2);
  // CHECK: sycl::int2 int2_c = sycl::int2(int2_b);
  int2 int2_c = int2(int2_b);
  // CHECK: sycl::int2 int2_d(int2_c);
  int2 int2_d(int2_c);
  // CHECK: func3_int2(int2_b, sycl::int2(int2_b), (sycl::int2)int2_b);
  func3_int2(int2_b, int2(int2_b), (int2)int2_b);
  // CHECK: sycl::int2 *int2_e;
  int2 *int2_e;
  // CHECK: sycl::int2 *int2_f;
  int2 *int2_f;
  // CHECK: int int2_g = int2_c.x();
  int int2_g = int2_c.x;
  // CHECK: int2_a.x() = int2_d.x();
  int2_a.x = int2_d.x;
  // CHECK: if (int2_b.x() == int2_d.x()) {}
  if (int2_b.x == int2_d.x) {}
  // CHECK: sycl::int2 int2_h[16];
  int2 int2_h[16];
  // CHECK: sycl::int2 int2_i[32];
  int2 int2_i[32];
  // CHECK: if (int2_h[12].x() == int2_i[12].x()) {}
  if (int2_h[12].x == int2_i[12].x) {}
  // CHECK: int2_f = (sycl::int2 *)int2_i;
  int2_f = (int2 *)int2_i;
  // CHECK: int2_a = (sycl::int2)int2_c;
  int2_a = (int2)int2_c;
  // CHECK: int2_b = sycl::int2(int2_b);
  int2_b = int2(int2_b);
  // CHECK: sycl::int2 int2_j, int2_k, int2_l, int2_m[16], *int2_n[32];
  int2 int2_j, int2_k, int2_l, int2_m[16], *int2_n[32];
  // CHECK: int int2_o = sizeof(sycl::int2);
  int int2_o = sizeof(int2);
  // CHECK: int int_p = sizeof(int);
  int int_p = sizeof(int);
  // CHECK: int int2_q = sizeof(int2_d);
  int int2_q = sizeof(int2_d);
  int *int2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::int2 *> int2_e_acc_ct0(int2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::int2 *> int2_cast_acc_ct1((sycl::int2 *)int2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_int2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_int2(int2_e_acc_ct0.get_raw_pointer(), int2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_int2<<<1,1>>>(int2_e, (int2 *)int2_cast);
  // CHECK: sycl::int2 int2_r = (sycl::int2){1,1};
  // CHECK-NEXT: auto int2_s = (sycl::int2){1,1};
  int2 int2_r = (int2){1,1};
  auto int2_s = (int2){1,1};
  return 0;
}

// CHECK: void func3_int3(sycl::int3 a, sycl::int3 b, sycl::int3 c) {
void func3_int3(int3 a, int3 b, int3 c) {
}
// CHECK: void func_int3(sycl::int3 a) {
void func_int3(int3 a) {
}
// CHECK: void kernel_int3(sycl::int3 *a, sycl::int3 *b) {
__global__ void kernel_int3(int3 *a, int3 *b) {
}

int main_int3() {
  // range default constructor does the right thing.
  // CHECK: sycl::int3 int3_a;
  int3 int3_a;
  // CHECK: sycl::int3 int3_b = sycl::int3(1, 2, 3);
  int3 int3_b = make_int3(1, 2, 3);
  // CHECK: sycl::int3 int3_c = sycl::int3(int3_b);
  int3 int3_c = int3(int3_b);
  // CHECK: sycl::int3 int3_d(int3_c);
  int3 int3_d(int3_c);
  // CHECK: func3_int3(int3_b, sycl::int3(int3_b), (sycl::int3)int3_b);
  func3_int3(int3_b, int3(int3_b), (int3)int3_b);
  // CHECK: sycl::int3 *int3_e;
  int3 *int3_e;
  // CHECK: sycl::int3 *int3_f;
  int3 *int3_f;
  // CHECK: int int3_g = int3_c.x();
  int int3_g = int3_c.x;
  // CHECK: int3_a.x() = int3_d.x();
  int3_a.x = int3_d.x;
  // CHECK: if (int3_b.x() == int3_d.x()) {}
  if (int3_b.x == int3_d.x) {}
  // CHECK: sycl::int3 int3_h[16];
  int3 int3_h[16];
  // CHECK: sycl::int3 int3_i[32];
  int3 int3_i[32];
  // CHECK: if (int3_h[12].x() == int3_i[12].x()) {}
  if (int3_h[12].x == int3_i[12].x) {}
  // CHECK: int3_f = (sycl::int3 *)int3_i;
  int3_f = (int3 *)int3_i;
  // CHECK: int3_a = (sycl::int3)int3_c;
  int3_a = (int3)int3_c;
  // CHECK: int3_b = sycl::int3(int3_b);
  int3_b = int3(int3_b);
  // CHECK: sycl::int3 int3_j, int3_k, int3_l, int3_m[16], *int3_n[32];
  int3 int3_j, int3_k, int3_l, int3_m[16], *int3_n[32];
  // CHECK: int int3_o = sizeof(sycl::int3);
  int int3_o = sizeof(int3);
  // CHECK: int int_p = sizeof(int);
  int int_p = sizeof(int);
  // CHECK: int int3_q = sizeof(int3_d);
  int int3_q = sizeof(int3_d);
  int *int3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::int3 *> int3_e_acc_ct0(int3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::int3 *> int3_cast_acc_ct1((sycl::int3 *)int3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_int3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_int3(int3_e_acc_ct0.get_raw_pointer(), int3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_int3<<<1,1>>>(int3_e, (int3 *)int3_cast);
  // CHECK: sycl::int3 int3_r = (sycl::int3){1,1,1};
  // CHECK-NEXT: auto int3_s = (sycl::int3){1,1,1};
  int3 int3_r = (int3){1,1,1};
  auto int3_s = (int3){1,1,1};
  return 0;
}

// CHECK: void func3_int4(sycl::int4 a, sycl::int4 b, sycl::int4 c) {
void func3_int4(int4 a, int4 b, int4 c) {
}
// CHECK: void func_int4(sycl::int4 a) {
void func_int4(int4 a) {
}
// CHECK: void kernel_int4(sycl::int4 *a, sycl::int4 *b) {
__global__ void kernel_int4(int4 *a, int4 *b) {
}

int main_int4() {
  // range default constructor does the right thing.
  // CHECK: sycl::int4 int4_a;
  int4 int4_a;
  // CHECK: sycl::int4 int4_b = sycl::int4(1, 2, 3, 4);
  int4 int4_b = make_int4(1, 2, 3, 4);
  // CHECK: sycl::int4 int4_c = sycl::int4(int4_b);
  int4 int4_c = int4(int4_b);
  // CHECK: sycl::int4 int4_d(int4_c);
  int4 int4_d(int4_c);
  // CHECK: func3_int4(int4_b, sycl::int4(int4_b), (sycl::int4)int4_b);
  func3_int4(int4_b, int4(int4_b), (int4)int4_b);
  // CHECK: sycl::int4 *int4_e;
  int4 *int4_e;
  // CHECK: sycl::int4 *int4_f;
  int4 *int4_f;
  // Check: int4_f->x() = int4_e->x();
  int4_f->x = int4_e->x;
  // CHECK: int int4_g = int4_c.x();
  int int4_g = int4_c.x;
  // CHECK: int4_a.x() = int4_d.x();
  int4_a.x = int4_d.x;
  // CHECK: if (int4_b.x() == int4_d.x()) {}
  if (int4_b.x == int4_d.x) {}
  // CHECK: sycl::int4 int4_h[16];
  int4 int4_h[16];
  // CHECK: sycl::int4 int4_i[32];
  int4 int4_i[32];
  // CHECK: if (int4_h[12].x() == int4_i[12].x()) {}
  if (int4_h[12].x == int4_i[12].x) {}
  // CHECK: int4_f = (sycl::int4 *)int4_i;
  int4_f = (int4 *)int4_i;
  // CHECK: int4_a = (sycl::int4)int4_c;
  int4_a = (int4)int4_c;
  // CHECK: int4_b = sycl::int4(int4_b);
  int4_b = int4(int4_b);
  // CHECK: sycl::int4 int4_j, int4_k, int4_l, int4_m[16], *int4_n[32];
  int4 int4_j, int4_k, int4_l, int4_m[16], *int4_n[32];
  // CHECK: int int4_o = sizeof(sycl::int4);
  int int4_o = sizeof(int4);
  // CHECK: int int_p = sizeof(int);
  int int_p = sizeof(int);
  // CHECK: int int4_q = sizeof(int4_d);
  int int4_q = sizeof(int4_d);
  int *int4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::int4 *> int4_e_acc_ct0(int4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::int4 *> int4_cast_acc_ct1((sycl::int4 *)int4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_int4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_int4(int4_e_acc_ct0.get_raw_pointer(), int4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_int4<<<1,1>>>(int4_e, (int4 *)int4_cast);
  // CHECK: sycl::int4 int4_r = (sycl::int4){1,1,1,1};
  // CHECK-NEXT: auto int4_s = (sycl::int4){1,1,1,1};
  int4 int4_r = (int4){1,1,1,1};
  auto int4_s = (int4){1,1,1,1};
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
  // CHECK: long long1_g = long1_c;
  long long1_g = long1_c.x;
  // CHECK: long1_a = long1_d;
  long1_a.x = long1_d.x;
  // CHECK: if (long1_b == long1_d) {}
  if (long1_b.x == long1_d.x) {}
  // CHECK: long long1_h[16];
  long1 long1_h[16];
  // CHECK: long long1_i[32];
  long1 long1_i[32];
  // CHECK: if (long1_h[12] == long1_i[12]) {}
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
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<long *> long1_e_acc_ct0(long1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<long *> long1_cast_acc_ct1((long *)long1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_long1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_long1(long1_e_acc_ct0.get_raw_pointer(), long1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_long1<<<1,1>>>(long1_e, (long1 *)long1_cast);
  // CHECK: long long1_r = (long){1};
  // CHECK-NEXT: auto long1_s = (long){1};
  long1 long1_r = (long1){1};
  auto long1_s = (long1){1};
  return 0;
}

// CHECK: void func3_long2(sycl::long2 a, sycl::long2 b, sycl::long2 c) {
void func3_long2(long2 a, long2 b, long2 c) {
}
// CHECK: void func_long2(sycl::long2 a) {
void func_long2(long2 a) {
}
// CHECK: void kernel_long2(sycl::long2 *a, sycl::long2 *b) {
__global__ void kernel_long2(long2 *a, long2 *b) {
}

int main_long2() {
  // range default constructor does the right thing.
  // CHECK: sycl::long2 long2_a;
  long2 long2_a;
  // CHECK: sycl::long2 long2_b = sycl::long2(1, 2);
  long2 long2_b = make_long2(1, 2);
  // CHECK: sycl::long2 long2_c = sycl::long2(long2_b);
  long2 long2_c = long2(long2_b);
  // CHECK: sycl::long2 long2_d(long2_c);
  long2 long2_d(long2_c);
  // CHECK: func3_long2(long2_b, sycl::long2(long2_b), (sycl::long2)long2_b);
  func3_long2(long2_b, long2(long2_b), (long2)long2_b);
  // CHECK: sycl::long2 *long2_e;
  long2 *long2_e;
  // CHECK: sycl::long2 *long2_f;
  long2 *long2_f;
  // CHECK: long long2_g = long2_c.x();
  long long2_g = long2_c.x;
  // CHECK: long2_a.x() = long2_d.x();
  long2_a.x = long2_d.x;
  // CHECK: if (long2_b.x() == long2_d.x()) {}
  if (long2_b.x == long2_d.x) {}
  // CHECK: sycl::long2 long2_h[16];
  long2 long2_h[16];
  // CHECK: sycl::long2 long2_i[32];
  long2 long2_i[32];
  // CHECK: if (long2_h[12].x() == long2_i[12].x()) {}
  if (long2_h[12].x == long2_i[12].x) {}
  // CHECK: long2_f = (sycl::long2 *)long2_i;
  long2_f = (long2 *)long2_i;
  // CHECK: long2_a = (sycl::long2)long2_c;
  long2_a = (long2)long2_c;
  // CHECK: long2_b = sycl::long2(long2_b);
  long2_b = long2(long2_b);
  // CHECK: sycl::long2 long2_j, long2_k, long2_l, long2_m[16], *long2_n[32];
  long2 long2_j, long2_k, long2_l, long2_m[16], *long2_n[32];
  // CHECK: int long2_o = sizeof(sycl::long2);
  int long2_o = sizeof(long2);
  // CHECK: int long_p = sizeof(long);
  int long_p = sizeof(long);
  // CHECK: int long2_q = sizeof(long2_d);
  int long2_q = sizeof(long2_d);
  int *long2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::long2 *> long2_e_acc_ct0(long2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::long2 *> long2_cast_acc_ct1((sycl::long2 *)long2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_long2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_long2(long2_e_acc_ct0.get_raw_pointer(), long2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_long2<<<1,1>>>(long2_e, (long2 *)long2_cast);
  // CHECK: sycl::long2 long2_r = (sycl::long2){1,1};
  // CHECK-NEXT: auto long2_s = (sycl::long2){1,1};
  long2 long2_r = (long2){1,1};
  auto long2_s = (long2){1,1};
  return 0;
}

// CHECK: void func3_long3(sycl::long3 a, sycl::long3 b, sycl::long3 c) {
void func3_long3(long3 a, long3 b, long3 c) {
}
// CHECK: void func_long3(sycl::long3 a) {
void func_long3(long3 a) {
}
// CHECK: void kernel_long3(sycl::long3 *a, sycl::long3 *b) {
__global__ void kernel_long3(long3 *a, long3 *b) {
}

int main_long3() {
  // range default constructor does the right thing.
  // CHECK: sycl::long3 long3_a;
  long3 long3_a;
  // CHECK: sycl::long3 long3_b = sycl::long3(1, 2, 3);
  long3 long3_b = make_long3(1, 2, 3);
  // CHECK: sycl::long3 long3_c = sycl::long3(long3_b);
  long3 long3_c = long3(long3_b);
  // CHECK: sycl::long3 long3_d(long3_c);
  long3 long3_d(long3_c);
  // CHECK: func3_long3(long3_b, sycl::long3(long3_b), (sycl::long3)long3_b);
  func3_long3(long3_b, long3(long3_b), (long3)long3_b);
  // CHECK: sycl::long3 *long3_e;
  long3 *long3_e;
  // CHECK: sycl::long3 *long3_f;
  long3 *long3_f;
  // CHECK: long long3_g = long3_c.x();
  long long3_g = long3_c.x;
  // CHECK: long3_a.x() = long3_d.x();
  long3_a.x = long3_d.x;
  // CHECK: if (long3_b.x() == long3_d.x()) {}
  if (long3_b.x == long3_d.x) {}
  // CHECK: sycl::long3 long3_h[16];
  long3 long3_h[16];
  // CHECK: sycl::long3 long3_i[32];
  long3 long3_i[32];
  // CHECK: if (long3_h[12].x() == long3_i[12].x()) {}
  if (long3_h[12].x == long3_i[12].x) {}
  // CHECK: long3_f = (sycl::long3 *)long3_i;
  long3_f = (long3 *)long3_i;
  // CHECK: long3_a = (sycl::long3)long3_c;
  long3_a = (long3)long3_c;
  // CHECK: long3_b = sycl::long3(long3_b);
  long3_b = long3(long3_b);
  // CHECK: sycl::long3 long3_j, long3_k, long3_l, long3_m[16], *long3_n[32];
  long3 long3_j, long3_k, long3_l, long3_m[16], *long3_n[32];
  // CHECK: int long3_o = sizeof(sycl::long3);
  int long3_o = sizeof(long3);
  // CHECK: int long_p = sizeof(long);
  int long_p = sizeof(long);
  // CHECK: int long3_q = sizeof(long3_d);
  int long3_q = sizeof(long3_d);
  int *long3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::long3 *> long3_e_acc_ct0(long3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::long3 *> long3_cast_acc_ct1((sycl::long3 *)long3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_long3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_long3(long3_e_acc_ct0.get_raw_pointer(), long3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_long3<<<1,1>>>(long3_e, (long3 *)long3_cast);
  // CHECK: sycl::long3 long3_r = (sycl::long3){1,1,1};
  // CHECK-NEXT: auto long3_s = (sycl::long3){1,1,1};
  long3 long3_r = (long3){1,1,1};
  auto long3_s = (long3){1,1,1};
  return 0;
}

// CHECK: void func3_long4(sycl::long4 a, sycl::long4 b, sycl::long4 c) {
void func3_long4(long4 a, long4 b, long4 c) {
}
// CHECK: void func_long4(sycl::long4 a) {
void func_long4(long4 a) {
}
// CHECK: void kernel_long4(sycl::long4 *a, sycl::long4 *b) {
__global__ void kernel_long4(long4 *a, long4 *b) {
}

int main_long4() {
  // range default constructor does the right thing.
  // CHECK: sycl::long4 long4_a;
  long4 long4_a;
  // CHECK: sycl::long4 long4_b = sycl::long4(1, 2, 3, 4);
  long4 long4_b = make_long4(1, 2, 3, 4);
  // CHECK: sycl::long4 long4_c = sycl::long4(long4_b);
  long4 long4_c = long4(long4_b);
  // CHECK: sycl::long4 long4_d(long4_c);
  long4 long4_d(long4_c);
  // CHECK: func3_long4(long4_b, sycl::long4(long4_b), (sycl::long4)long4_b);
  func3_long4(long4_b, long4(long4_b), (long4)long4_b);
  // CHECK: sycl::long4 *long4_e;
  long4 *long4_e;
  // CHECK: sycl::long4 *long4_f;
  long4 *long4_f;
  // CHECK: long long4_g = long4_c.x();
  long long4_g = long4_c.x;
  // CHECK: long4_a.x() = long4_d.x();
  long4_a.x = long4_d.x;
  // CHECK: if (long4_b.x() == long4_d.x()) {}
  if (long4_b.x == long4_d.x) {}
  // CHECK: sycl::long4 long4_h[16];
  long4 long4_h[16];
  // CHECK: sycl::long4 long4_i[32];
  long4 long4_i[32];
  // CHECK: if (long4_h[12].x() == long4_i[12].x()) {}
  if (long4_h[12].x == long4_i[12].x) {}
  // CHECK: long4_f = (sycl::long4 *)long4_i;
  long4_f = (long4 *)long4_i;
  // CHECK: long4_a = (sycl::long4)long4_c;
  long4_a = (long4)long4_c;
  // CHECK: long4_b = sycl::long4(long4_b);
  long4_b = long4(long4_b);
  // CHECK: sycl::long4 long4_j, long4_k, long4_l, long4_m[16], *long4_n[32];
  long4 long4_j, long4_k, long4_l, long4_m[16], *long4_n[32];
  // CHECK: int long4_o = sizeof(sycl::long4);
  int long4_o = sizeof(long4);
  // CHECK: int long_p = sizeof(long);
  int long_p = sizeof(long);
  // CHECK: int long4_q = sizeof(long4_d);
  int long4_q = sizeof(long4_d);
  int *long4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::long4 *> long4_e_acc_ct0(long4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::long4 *> long4_cast_acc_ct1((sycl::long4 *)long4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_long4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_long4(long4_e_acc_ct0.get_raw_pointer(), long4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_long4<<<1,1>>>(long4_e, (long4 *)long4_cast);
  // CHECK: sycl::long4 long4_r = (sycl::long4){1,1,1,1};
  // CHECK-NEXT: auto long4_s = (sycl::long4){1,1,1,1};
  long4 long4_r = (long4){1,1,1,1};
  auto long4_s = (long4){1,1,1,1};
  return 0;
}

// CHECK: void func3_longlong1(int64_t a, int64_t b, int64_t c) {
void func3_longlong1(longlong1 a, longlong1 b, longlong1 c) {
}
// CHECK: void func_longlong1(int64_t a) {
void func_longlong1(longlong1 a) {
}
// CHECK: void kernel_longlong1(int64_t *a, int64_t *b) {
__global__ void kernel_longlong1(longlong1 *a, longlong1 *b) {
}

int main_longlong1() {
  // range default constructor does the right thing.
  // CHECK: int64_t longlong1_a;
  longlong1 longlong1_a;
  // CHECK: int64_t longlong1_b = int64_t(1);
  longlong1 longlong1_b = make_longlong1(1);
  // CHECK: int64_t longlong1_c = int64_t(longlong1_b);
  longlong1 longlong1_c = longlong1(longlong1_b);
  // CHECK: int64_t longlong1_d(longlong1_c);
  longlong1 longlong1_d(longlong1_c);
  // CHECK: func3_longlong1(longlong1_b, int64_t(longlong1_b), (int64_t)longlong1_b);
  func3_longlong1(longlong1_b, longlong1(longlong1_b), (longlong1)longlong1_b);
  // CHECK: int64_t *longlong1_e;
  longlong1 *longlong1_e;
  // CHECK: int64_t *longlong1_f;
  longlong1 *longlong1_f;
  // CHECK: long long longlong1_g = longlong1_c;
  long long longlong1_g = longlong1_c.x;
  // CHECK: longlong1_a = longlong1_d;
  longlong1_a.x = longlong1_d.x;
  // CHECK: if (longlong1_b == longlong1_d) {}
  if (longlong1_b.x == longlong1_d.x) {}
  // CHECK: int64_t longlong1_h[16];
  longlong1 longlong1_h[16];
  // CHECK: int64_t longlong1_i[32];
  longlong1 longlong1_i[32];
  // CHECK: if (longlong1_h[12] == longlong1_i[12]) {}
  if (longlong1_h[12].x == longlong1_i[12].x) {}
  // CHECK: longlong1_f = (int64_t *)longlong1_i;
  longlong1_f = (longlong1 *)longlong1_i;
  // CHECK: longlong1_a = (int64_t)longlong1_c;
  longlong1_a = (longlong1)longlong1_c;
  // CHECK: longlong1_b = int64_t(longlong1_b);
  longlong1_b = longlong1(longlong1_b);
  // CHECK: int64_t longlong1_j, longlong1_k, longlong1_l, longlong1_m[16], *longlong1_n[32];
  longlong1 longlong1_j, longlong1_k, longlong1_l, longlong1_m[16], *longlong1_n[32];
  // CHECK: int longlong1_o = sizeof(int64_t);
  int longlong1_o = sizeof(longlong1);
  // CHECK: int long long_p = sizeof(long long);
  int long long_p = sizeof(long long);
  // CHECK: int longlong1_q = sizeof(longlong1_d);
  int longlong1_q = sizeof(longlong1_d);
  int *longlong1_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<int64_t *> longlong1_e_acc_ct0(longlong1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<int64_t *> longlong1_cast_acc_ct1((int64_t *)longlong1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_longlong1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_longlong1(longlong1_e_acc_ct0.get_raw_pointer(), longlong1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_longlong1<<<1,1>>>(longlong1_e, (longlong1 *)longlong1_cast);
  // CHECK: int64_t longlong1_r = (int64_t){1};
  // CHECK-NEXT: auto longlong1_s = (int64_t){1};
  longlong1 longlong1_r = (longlong1){1};
  auto longlong1_s = (longlong1){1};
  return 0;
}

// CHECK: void func3_longlong2(sycl::longlong2 a, sycl::longlong2 b, sycl::longlong2 c) {
void func3_longlong2(longlong2 a, longlong2 b, longlong2 c) {
}
// CHECK: void func_longlong2(sycl::longlong2 a) {
void func_longlong2(longlong2 a) {
}
// CHECK: void kernel_longlong2(sycl::longlong2 *a, sycl::longlong2 *b) {
__global__ void kernel_longlong2(longlong2 *a, longlong2 *b) {
}

int main_longlong2() {
  // range default constructor does the right thing.
  // CHECK: sycl::longlong2 longlong2_a;
  longlong2 longlong2_a;
  // CHECK: sycl::longlong2 longlong2_b = sycl::longlong2(1, 2);
  longlong2 longlong2_b = make_longlong2(1, 2);
  // CHECK: sycl::longlong2 longlong2_c = sycl::longlong2(longlong2_b);
  longlong2 longlong2_c = longlong2(longlong2_b);
  // CHECK: sycl::longlong2 longlong2_d(longlong2_c);
  longlong2 longlong2_d(longlong2_c);
  // CHECK: func3_longlong2(longlong2_b, sycl::longlong2(longlong2_b), (sycl::longlong2)longlong2_b);
  func3_longlong2(longlong2_b, longlong2(longlong2_b), (longlong2)longlong2_b);
  // CHECK: sycl::longlong2 *longlong2_e;
  longlong2 *longlong2_e;
  // CHECK: sycl::longlong2 *longlong2_f;
  longlong2 *longlong2_f;
  // CHECK: long long longlong2_g = longlong2_c.x();
  long long longlong2_g = longlong2_c.x;
  // CHECK: longlong2_a.x() = longlong2_d.x();
  longlong2_a.x = longlong2_d.x;
  // CHECK: if (longlong2_b.x() == longlong2_d.x()) {}
  if (longlong2_b.x == longlong2_d.x) {}
  // CHECK: sycl::longlong2 longlong2_h[16];
  longlong2 longlong2_h[16];
  // CHECK: sycl::longlong2 longlong2_i[32];
  longlong2 longlong2_i[32];
  // CHECK: if (longlong2_h[12].x() == longlong2_i[12].x()) {}
  if (longlong2_h[12].x == longlong2_i[12].x) {}
  // CHECK: longlong2_f = (sycl::longlong2 *)longlong2_i;
  longlong2_f = (longlong2 *)longlong2_i;
  // CHECK: longlong2_a = (sycl::longlong2)longlong2_c;
  longlong2_a = (longlong2)longlong2_c;
  // CHECK: longlong2_b = sycl::longlong2(longlong2_b);
  longlong2_b = longlong2(longlong2_b);
  // CHECK: sycl::longlong2 longlong2_j, longlong2_k, longlong2_l, longlong2_m[16], *longlong2_n[32];
  longlong2 longlong2_j, longlong2_k, longlong2_l, longlong2_m[16], *longlong2_n[32];
  // CHECK: int longlong2_o = sizeof(sycl::longlong2);
  int longlong2_o = sizeof(longlong2);
  // CHECK: int long long_p = sizeof(long long);
  int long long_p = sizeof(long long);
  // CHECK: int longlong2_q = sizeof(longlong2_d);
  int longlong2_q = sizeof(longlong2_d);
  int *longlong2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::longlong2 *> longlong2_e_acc_ct0(longlong2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::longlong2 *> longlong2_cast_acc_ct1((sycl::longlong2 *)longlong2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_longlong2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_longlong2(longlong2_e_acc_ct0.get_raw_pointer(), longlong2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_longlong2<<<1,1>>>(longlong2_e, (longlong2 *)longlong2_cast);
  // CHECK: sycl::longlong2 longlong2_r = (sycl::longlong2){1,1};
  // CHECK-NEXT: auto longlong2_s = (sycl::longlong2){1,1};
  longlong2 longlong2_r = (longlong2){1,1};
  auto longlong2_s = (longlong2){1,1};
  return 0;
}

// CHECK: void func3_longlong3(sycl::longlong3 a, sycl::longlong3 b, sycl::longlong3 c) {
void func3_longlong3(longlong3 a, longlong3 b, longlong3 c) {
}
// CHECK: void func_longlong3(sycl::longlong3 a) {
void func_longlong3(longlong3 a) {
}
// CHECK: void kernel_longlong3(sycl::longlong3 *a, sycl::longlong3 *b) {
__global__ void kernel_longlong3(longlong3 *a, longlong3 *b) {
}

int main_longlong3() {
  // range default constructor does the right thing.
  // CHECK: sycl::longlong3 longlong3_a;
  longlong3 longlong3_a;
  // CHECK: sycl::longlong3 longlong3_b = sycl::longlong3(1, 2, 3);
  longlong3 longlong3_b = make_longlong3(1, 2, 3);
  // CHECK: sycl::longlong3 longlong3_c = sycl::longlong3(longlong3_b);
  longlong3 longlong3_c = longlong3(longlong3_b);
  // CHECK: sycl::longlong3 longlong3_d(longlong3_c);
  longlong3 longlong3_d(longlong3_c);
  // CHECK: func3_longlong3(longlong3_b, sycl::longlong3(longlong3_b), (sycl::longlong3)longlong3_b);
  func3_longlong3(longlong3_b, longlong3(longlong3_b), (longlong3)longlong3_b);
  // CHECK: sycl::longlong3 *longlong3_e;
  longlong3 *longlong3_e;
  // CHECK: sycl::longlong3 *longlong3_f;
  longlong3 *longlong3_f;
  // CHECK: long long longlong3_g = longlong3_c.x();
  long long longlong3_g = longlong3_c.x;
  // CHECK: longlong3_a.x() = longlong3_d.x();
  longlong3_a.x = longlong3_d.x;
  // CHECK: if (longlong3_b.x() == longlong3_d.x()) {}
  if (longlong3_b.x == longlong3_d.x) {}
  // CHECK: sycl::longlong3 longlong3_h[16];
  longlong3 longlong3_h[16];
  // CHECK: sycl::longlong3 longlong3_i[32];
  longlong3 longlong3_i[32];
  // CHECK: if (longlong3_h[12].x() == longlong3_i[12].x()) {}
  if (longlong3_h[12].x == longlong3_i[12].x) {}
  // CHECK: longlong3_f = (sycl::longlong3 *)longlong3_i;
  longlong3_f = (longlong3 *)longlong3_i;
  // CHECK: longlong3_a = (sycl::longlong3)longlong3_c;
  longlong3_a = (longlong3)longlong3_c;
  // CHECK: longlong3_b = sycl::longlong3(longlong3_b);
  longlong3_b = longlong3(longlong3_b);
  // CHECK: sycl::longlong3 longlong3_j, longlong3_k, longlong3_l, longlong3_m[16], *longlong3_n[32];
  longlong3 longlong3_j, longlong3_k, longlong3_l, longlong3_m[16], *longlong3_n[32];
  // CHECK: int longlong3_o = sizeof(sycl::longlong3);
  int longlong3_o = sizeof(longlong3);
  // CHECK: int long long_p = sizeof(long long);
  int long long_p = sizeof(long long);
  // CHECK: int longlong3_q = sizeof(longlong3_d);
  int longlong3_q = sizeof(longlong3_d);
  int *longlong3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::longlong3 *> longlong3_e_acc_ct0(longlong3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::longlong3 *> longlong3_cast_acc_ct1((sycl::longlong3 *)longlong3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_longlong3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_longlong3(longlong3_e_acc_ct0.get_raw_pointer(), longlong3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_longlong3<<<1,1>>>(longlong3_e, (longlong3 *)longlong3_cast);
  // CHECK: sycl::longlong3 longlong3_r = (sycl::longlong3){1,1,1};
  // CHECK-NEXT: auto longlong3_s = (sycl::longlong3){1,1,1};
  longlong3 longlong3_r = (longlong3){1,1,1};
  auto longlong3_s = (longlong3){1,1,1};
  return 0;
}

// CHECK: void func3_longlong4(sycl::longlong4 a, sycl::longlong4 b, sycl::longlong4 c) {
void func3_longlong4(longlong4 a, longlong4 b, longlong4 c) {
}
// CHECK: void func_longlong4(sycl::longlong4 a) {
void func_longlong4(longlong4 a) {
}
// CHECK: void kernel_longlong4(sycl::longlong4 *a, sycl::longlong4 *b) {
__global__ void kernel_longlong4(longlong4 *a, longlong4 *b) {
}

int main_longlong4() {
  // range default constructor does the right thing.
  // CHECK: sycl::longlong4 longlong4_a;
  longlong4 longlong4_a;
  // CHECK: sycl::longlong4 longlong4_b = sycl::longlong4(1, 2, 3, 4);
  longlong4 longlong4_b = make_longlong4(1, 2, 3, 4);
  // CHECK: sycl::longlong4 longlong4_c = sycl::longlong4(longlong4_b);
  longlong4 longlong4_c = longlong4(longlong4_b);
  // CHECK: sycl::longlong4 longlong4_d(longlong4_c);
  longlong4 longlong4_d(longlong4_c);
  // CHECK: func3_longlong4(longlong4_b, sycl::longlong4(longlong4_b), (sycl::longlong4)longlong4_b);
  func3_longlong4(longlong4_b, longlong4(longlong4_b), (longlong4)longlong4_b);
  // CHECK: sycl::longlong4 *longlong4_e;
  longlong4 *longlong4_e;
  // CHECK: sycl::longlong4 *longlong4_f;
  longlong4 *longlong4_f;
  // CHECK: long long longlong4_g = longlong4_c.x();
  long long longlong4_g = longlong4_c.x;
  // CHECK: longlong4_a.x() = longlong4_d.x();
  longlong4_a.x = longlong4_d.x;
  // CHECK: if (longlong4_b.x() == longlong4_d.x()) {}
  if (longlong4_b.x == longlong4_d.x) {}
  // CHECK: sycl::longlong4 longlong4_h[16];
  longlong4 longlong4_h[16];
  // CHECK: sycl::longlong4 longlong4_i[32];
  longlong4 longlong4_i[32];
  // CHECK: if (longlong4_h[12].x() == longlong4_i[12].x()) {}
  if (longlong4_h[12].x == longlong4_i[12].x) {}
  // CHECK: longlong4_f = (sycl::longlong4 *)longlong4_i;
  longlong4_f = (longlong4 *)longlong4_i;
  // CHECK: longlong4_a = (sycl::longlong4)longlong4_c;
  longlong4_a = (longlong4)longlong4_c;
  // CHECK: longlong4_b = sycl::longlong4(longlong4_b);
  longlong4_b = longlong4(longlong4_b);
  // CHECK: sycl::longlong4 longlong4_j, longlong4_k, longlong4_l, longlong4_m[16], *longlong4_n[32];
  longlong4 longlong4_j, longlong4_k, longlong4_l, longlong4_m[16], *longlong4_n[32];
  // CHECK: int longlong4_o = sizeof(sycl::longlong4);
  int longlong4_o = sizeof(longlong4);
  // CHECK: int long long_p = sizeof(long long);
  int long long_p = sizeof(long long);
  // CHECK: int longlong4_q = sizeof(longlong4_d);
  int longlong4_q = sizeof(longlong4_d);
  int *longlong4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::longlong4 *> longlong4_e_acc_ct0(longlong4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::longlong4 *> longlong4_cast_acc_ct1((sycl::longlong4 *)longlong4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_longlong4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_longlong4(longlong4_e_acc_ct0.get_raw_pointer(), longlong4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_longlong4<<<1,1>>>(longlong4_e, (longlong4 *)longlong4_cast);
  // CHECK: sycl::longlong4 longlong4_r = (sycl::longlong4){1,1,1,1};
  // CHECK-NEXT: auto longlong4_s = (sycl::longlong4){1,1,1,1};
  longlong4 longlong4_r = (longlong4){1,1,1,1};
  auto longlong4_s = (longlong4){1,1,1,1};
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
  // CHECK: short short1_g = short1_c;
  short short1_g = short1_c.x;
  // CHECK: short1_a = short1_d;
  short1_a.x = short1_d.x;
  // CHECK: if (short1_b == short1_d) {}
  if (short1_b.x == short1_d.x) {}
  // CHECK: short short1_h[16];
  short1 short1_h[16];
  // CHECK: short short1_i[32];
  short1 short1_i[32];
  // CHECK: if (short1_h[12] == short1_i[12]) {}
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
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<short *> short1_e_acc_ct0(short1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<short *> short1_cast_acc_ct1((short *)short1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_short1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_short1(short1_e_acc_ct0.get_raw_pointer(), short1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_short1<<<1,1>>>(short1_e, (short1 *)short1_cast);
  // CHECK: short short1_r = (short){1};
  // CHECK-NEXT: auto short1_s = (short){1};
  short1 short1_r = (short1){1};
  auto short1_s = (short1){1};
  return 0;
}

// CHECK: void func3_short2(sycl::short2 a, sycl::short2 b, sycl::short2 c) {
void func3_short2(short2 a, short2 b, short2 c) {
}
// CHECK: void func_short2(sycl::short2 a) {
void func_short2(short2 a) {
}
// CHECK: void kernel_short2(sycl::short2 *a, sycl::short2 *b) {
__global__ void kernel_short2(short2 *a, short2 *b) {
}

int main_short2() {
  // range default constructor does the right thing.
  // CHECK: sycl::short2 short2_a;
  short2 short2_a;
  // CHECK: sycl::short2 short2_b = sycl::short2(1, 2);
  short2 short2_b = make_short2(1, 2);
  // CHECK: sycl::short2 short2_c = sycl::short2(short2_b);
  short2 short2_c = short2(short2_b);
  // CHECK: sycl::short2 short2_d(short2_c);
  short2 short2_d(short2_c);
  // CHECK: func3_short2(short2_b, sycl::short2(short2_b), (sycl::short2)short2_b);
  func3_short2(short2_b, short2(short2_b), (short2)short2_b);
  // CHECK: sycl::short2 *short2_e;
  short2 *short2_e;
  // CHECK: sycl::short2 *short2_f;
  short2 *short2_f;
  // CHECK: short short2_g = short2_c.x();
  short short2_g = short2_c.x;
  // CHECK: short2_a.x() = short2_d.x();
  short2_a.x = short2_d.x;
  // CHECK: if (short2_b.x() == short2_d.x()) {}
  if (short2_b.x == short2_d.x) {}
  // CHECK: sycl::short2 short2_h[16];
  short2 short2_h[16];
  // CHECK: sycl::short2 short2_i[32];
  short2 short2_i[32];
  // CHECK: if (short2_h[12].x() == short2_i[12].x()) {}
  if (short2_h[12].x == short2_i[12].x) {}
  // CHECK: short2_f = (sycl::short2 *)short2_i;
  short2_f = (short2 *)short2_i;
  // CHECK: short2_a = (sycl::short2)short2_c;
  short2_a = (short2)short2_c;
  // CHECK: short2_b = sycl::short2(short2_b);
  short2_b = short2(short2_b);
  // CHECK: sycl::short2 short2_j, short2_k, short2_l, short2_m[16], *short2_n[32];
  short2 short2_j, short2_k, short2_l, short2_m[16], *short2_n[32];
  // CHECK: int short2_o = sizeof(sycl::short2);
  int short2_o = sizeof(short2);
  // CHECK: int short_p = sizeof(short);
  int short_p = sizeof(short);
  // CHECK: int short2_q = sizeof(short2_d);
  int short2_q = sizeof(short2_d);
  int *short2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::short2 *> short2_e_acc_ct0(short2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::short2 *> short2_cast_acc_ct1((sycl::short2 *)short2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_short2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_short2(short2_e_acc_ct0.get_raw_pointer(), short2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_short2<<<1,1>>>(short2_e, (short2 *)short2_cast);
  // CHECK: sycl::short2 short2_r = (sycl::short2){1,1};
  // CHECK-NEXT: auto short2_s = (sycl::short2){1,1};
  short2 short2_r = (short2){1,1};
  auto short2_s = (short2){1,1};
  return 0;
}

// CHECK: void func3_short3(sycl::short3 a, sycl::short3 b, sycl::short3 c) {
void func3_short3(short3 a, short3 b, short3 c) {
}
// CHECK: void func_short3(sycl::short3 a) {
void func_short3(short3 a) {
}
// CHECK: void kernel_short3(sycl::short3 *a, sycl::short3 *b) {
__global__ void kernel_short3(short3 *a, short3 *b) {
}

int main_short3() {
  // range default constructor does the right thing.
  // CHECK: sycl::short3 short3_a;
  short3 short3_a;
  // CHECK: sycl::short3 short3_b = sycl::short3(1, 2, 3);
  short3 short3_b = make_short3(1, 2, 3);
  // CHECK: sycl::short3 short3_c = sycl::short3(short3_b);
  short3 short3_c = short3(short3_b);
  // CHECK: sycl::short3 short3_d(short3_c);
  short3 short3_d(short3_c);
  // CHECK: func3_short3(short3_b, sycl::short3(short3_b), (sycl::short3)short3_b);
  func3_short3(short3_b, short3(short3_b), (short3)short3_b);
  // CHECK: sycl::short3 *short3_e;
  short3 *short3_e;
  // CHECK: sycl::short3 *short3_f;
  short3 *short3_f;
  // CHECK: short short3_g = short3_c.x();
  short short3_g = short3_c.x;
  // CHECK: short3_a.x() = short3_d.x();
  short3_a.x = short3_d.x;
  // CHECK: if (short3_b.x() == short3_d.x()) {}
  if (short3_b.x == short3_d.x) {}
  // CHECK: sycl::short3 short3_h[16];
  short3 short3_h[16];
  // CHECK: sycl::short3 short3_i[32];
  short3 short3_i[32];
  // CHECK: if (short3_h[12].x() == short3_i[12].x()) {}
  if (short3_h[12].x == short3_i[12].x) {}
  // CHECK: short3_f = (sycl::short3 *)short3_i;
  short3_f = (short3 *)short3_i;
  // CHECK: short3_a = (sycl::short3)short3_c;
  short3_a = (short3)short3_c;
  // CHECK: short3_b = sycl::short3(short3_b);
  short3_b = short3(short3_b);
  // CHECK: sycl::short3 short3_j, short3_k, short3_l, short3_m[16], *short3_n[32];
  short3 short3_j, short3_k, short3_l, short3_m[16], *short3_n[32];
  // CHECK: int short3_o = sizeof(sycl::short3);
  int short3_o = sizeof(short3);
  // CHECK: int short_p = sizeof(short);
  int short_p = sizeof(short);
  // CHECK: int short3_q = sizeof(short3_d);
  int short3_q = sizeof(short3_d);
  int *short3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::short3 *> short3_e_acc_ct0(short3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::short3 *> short3_cast_acc_ct1((sycl::short3 *)short3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_short3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_short3(short3_e_acc_ct0.get_raw_pointer(), short3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_short3<<<1,1>>>(short3_e, (short3 *)short3_cast);
  // CHECK: sycl::short3 short3_r = (sycl::short3){1,1,1};
  // CHECK-NEXT: auto short3_s = (sycl::short3){1,1,1};
  short3 short3_r = (short3){1,1,1};
  auto short3_s = (short3){1,1,1};
  return 0;
}

// CHECK: void func3_short4(sycl::short4 a, sycl::short4 b, sycl::short4 c) {
void func3_short4(short4 a, short4 b, short4 c) {
}
// CHECK: void func_short4(sycl::short4 a) {
void func_short4(short4 a) {
}
// CHECK: void kernel_short4(sycl::short4 *a, sycl::short4 *b) {
__global__ void kernel_short4(short4 *a, short4 *b) {
}

int main_short4() {
  // range default constructor does the right thing.
  // CHECK: sycl::short4 short4_a;
  short4 short4_a;
  // CHECK: sycl::short4 short4_b = sycl::short4(1, 2, 3, 4);
  short4 short4_b = make_short4(1, 2, 3, 4);
  // CHECK: sycl::short4 short4_c = sycl::short4(short4_b);
  short4 short4_c = short4(short4_b);
  // CHECK: sycl::short4 short4_d(short4_c);
  short4 short4_d(short4_c);
  // CHECK: func3_short4(short4_b, sycl::short4(short4_b), (sycl::short4)short4_b);
  func3_short4(short4_b, short4(short4_b), (short4)short4_b);
  // CHECK: sycl::short4 *short4_e;
  short4 *short4_e;
  // CHECK: sycl::short4 *short4_f;
  short4 *short4_f;
  // CHECK: short short4_g = short4_c.x();
  short short4_g = short4_c.x;
  // CHECK: short4_a.x() = short4_d.x();
  short4_a.x = short4_d.x;
  // CHECK: if (short4_b.x() == short4_d.x()) {}
  if (short4_b.x == short4_d.x) {}
  // CHECK: sycl::short4 short4_h[16];
  short4 short4_h[16];
  // CHECK: sycl::short4 short4_i[32];
  short4 short4_i[32];
  // CHECK: if (short4_h[12].x() == short4_i[12].x()) {}
  if (short4_h[12].x == short4_i[12].x) {}
  // CHECK: short4_f = (sycl::short4 *)short4_i;
  short4_f = (short4 *)short4_i;
  // CHECK: short4_a = (sycl::short4)short4_c;
  short4_a = (short4)short4_c;
  // CHECK: short4_b = sycl::short4(short4_b);
  short4_b = short4(short4_b);
  // CHECK: sycl::short4 short4_j, short4_k, short4_l, short4_m[16], *short4_n[32];
  short4 short4_j, short4_k, short4_l, short4_m[16], *short4_n[32];
  // CHECK: int short4_o = sizeof(sycl::short4);
  int short4_o = sizeof(short4);
  // CHECK: int short_p = sizeof(short);
  int short_p = sizeof(short);
  // CHECK: int short4_q = sizeof(short4_d);
  int short4_q = sizeof(short4_d);
  int *short4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::short4 *> short4_e_acc_ct0(short4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::short4 *> short4_cast_acc_ct1((sycl::short4 *)short4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_short4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_short4(short4_e_acc_ct0.get_raw_pointer(), short4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_short4<<<1,1>>>(short4_e, (short4 *)short4_cast);
  // CHECK: sycl::short4 short4_r = (sycl::short4){1,1,1,1};
  // CHECK-NEXT: auto short4_s = (sycl::short4){1,1,1,1};
  short4 short4_r = (short4){1,1,1,1};
  auto short4_s = (short4){1,1,1,1};
  return 0;
}

// CHECK: void func3_uchar1(uint8_t a, uint8_t b, uint8_t c) {
void func3_uchar1(uchar1 a, uchar1 b, uchar1 c) {
}
// CHECK: void func_uchar1(uint8_t a) {
void func_uchar1(uchar1 a) {
}
// CHECK: void kernel_uchar1(uint8_t *a, uint8_t *b) {
__global__ void kernel_uchar1(uchar1 *a, uchar1 *b) {
}

int main_uchar1() {
  // range default constructor does the right thing.
  // CHECK: uint8_t uchar1_a;
  uchar1 uchar1_a;
  // CHECK: uint8_t uchar1_b = uint8_t(1);
  uchar1 uchar1_b = make_uchar1(1);
  // CHECK: uint8_t uchar1_c = uint8_t(uchar1_b);
  uchar1 uchar1_c = uchar1(uchar1_b);
  // CHECK: uint8_t uchar1_d(uchar1_c);
  uchar1 uchar1_d(uchar1_c);
  // CHECK: func3_uchar1(uchar1_b, uint8_t(uchar1_b), (uint8_t)uchar1_b);
  func3_uchar1(uchar1_b, uchar1(uchar1_b), (uchar1)uchar1_b);
  // CHECK: uint8_t *uchar1_e;
  uchar1 *uchar1_e;
  // CHECK: uint8_t *uchar1_f;
  uchar1 *uchar1_f;
  // CHECK: unsigned char uchar1_g = uchar1_c;
  unsigned char uchar1_g = uchar1_c.x;
  // CHECK: uchar1_a = uchar1_d;
  uchar1_a.x = uchar1_d.x;
  // CHECK: if (uchar1_b == uchar1_d) {}
  if (uchar1_b.x == uchar1_d.x) {}
  // CHECK: uint8_t uchar1_h[16];
  uchar1 uchar1_h[16];
  // CHECK: uint8_t uchar1_i[32];
  uchar1 uchar1_i[32];
  // CHECK: if (uchar1_h[12] == uchar1_i[12]) {}
  if (uchar1_h[12].x == uchar1_i[12].x) {}
  // CHECK: uchar1_f = (uint8_t *)uchar1_i;
  uchar1_f = (uchar1 *)uchar1_i;
  // CHECK: uchar1_a = (uint8_t)uchar1_c;
  uchar1_a = (uchar1)uchar1_c;
  // CHECK: uchar1_b = uint8_t(uchar1_b);
  uchar1_b = uchar1(uchar1_b);
  // CHECK: uint8_t uchar1_j, uchar1_k, uchar1_l, uchar1_m[16], *uchar1_n[32];
  uchar1 uchar1_j, uchar1_k, uchar1_l, uchar1_m[16], *uchar1_n[32];
  // CHECK: int uchar1_o = sizeof(uint8_t);
  int uchar1_o = sizeof(uchar1);
  // CHECK: int unsigned char_p = sizeof(unsigned char);
  int unsigned char_p = sizeof(unsigned char);
  // CHECK: int uchar1_q = sizeof(uchar1_d);
  int uchar1_q = sizeof(uchar1_d);
  int *uchar1_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<uint8_t *> uchar1_e_acc_ct0(uchar1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<uint8_t *> uchar1_cast_acc_ct1((uint8_t *)uchar1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_uchar1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_uchar1(uchar1_e_acc_ct0.get_raw_pointer(), uchar1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_uchar1<<<1,1>>>(uchar1_e, (uchar1 *)uchar1_cast);
  // CHECK: uint8_t uchar1_r = (uint8_t){1};
  // CHECK-NEXT: auto uchar1_s = (uint8_t){1};
  uchar1 uchar1_r = (uchar1){1};
  auto uchar1_s = (uchar1){1};
  return 0;
}

// CHECK: void func3_uchar2(sycl::uchar2 a, sycl::uchar2 b, sycl::uchar2 c) {
void func3_uchar2(uchar2 a, uchar2 b, uchar2 c) {
}
// CHECK: void func_uchar2(sycl::uchar2 a) {
void func_uchar2(uchar2 a) {
}
// CHECK: void kernel_uchar2(sycl::uchar2 *a, sycl::uchar2 *b) {
__global__ void kernel_uchar2(uchar2 *a, uchar2 *b) {
}

int main_uchar2() {
  // range default constructor does the right thing.
  // CHECK: sycl::uchar2 uchar2_a;
  uchar2 uchar2_a;
  // CHECK: sycl::uchar2 uchar2_b = sycl::uchar2(1, 2);
  uchar2 uchar2_b = make_uchar2(1, 2);
  // CHECK: sycl::uchar2 uchar2_c = sycl::uchar2(uchar2_b);
  uchar2 uchar2_c = uchar2(uchar2_b);
  // CHECK: sycl::uchar2 uchar2_d(uchar2_c);
  uchar2 uchar2_d(uchar2_c);
  // CHECK: func3_uchar2(uchar2_b, sycl::uchar2(uchar2_b), (sycl::uchar2)uchar2_b);
  func3_uchar2(uchar2_b, uchar2(uchar2_b), (uchar2)uchar2_b);
  // CHECK: sycl::uchar2 *uchar2_e;
  uchar2 *uchar2_e;
  // CHECK: sycl::uchar2 *uchar2_f;
  uchar2 *uchar2_f;
  // CHECK: unsigned char uchar2_g = uchar2_c.x();
  unsigned char uchar2_g = uchar2_c.x;
  // CHECK: uchar2_a.x() = uchar2_d.x();
  uchar2_a.x = uchar2_d.x;
  // CHECK: if (uchar2_b.x() == uchar2_d.x()) {}
  if (uchar2_b.x == uchar2_d.x) {}
  // CHECK: sycl::uchar2 uchar2_h[16];
  uchar2 uchar2_h[16];
  // CHECK: sycl::uchar2 uchar2_i[32];
  uchar2 uchar2_i[32];
  // CHECK: if (uchar2_h[12].x() == uchar2_i[12].x()) {}
  if (uchar2_h[12].x == uchar2_i[12].x) {}
  // CHECK: uchar2_f = (sycl::uchar2 *)uchar2_i;
  uchar2_f = (uchar2 *)uchar2_i;
  // CHECK: uchar2_a = (sycl::uchar2)uchar2_c;
  uchar2_a = (uchar2)uchar2_c;
  // CHECK: uchar2_b = sycl::uchar2(uchar2_b);
  uchar2_b = uchar2(uchar2_b);
  // CHECK: sycl::uchar2 uchar2_j, uchar2_k, uchar2_l, uchar2_m[16], *uchar2_n[32];
  uchar2 uchar2_j, uchar2_k, uchar2_l, uchar2_m[16], *uchar2_n[32];
  // CHECK: int uchar2_o = sizeof(sycl::uchar2);
  int uchar2_o = sizeof(uchar2);
  // CHECK: int unsigned char_p = sizeof(unsigned char);
  int unsigned char_p = sizeof(unsigned char);
  // CHECK: int uchar2_q = sizeof(uchar2_d);
  int uchar2_q = sizeof(uchar2_d);
  int *uchar2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uchar2 *> uchar2_e_acc_ct0(uchar2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uchar2 *> uchar2_cast_acc_ct1((sycl::uchar2 *)uchar2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_uchar2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_uchar2(uchar2_e_acc_ct0.get_raw_pointer(), uchar2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_uchar2<<<1,1>>>(uchar2_e, (uchar2 *)uchar2_cast);
  // CHECK: sycl::uchar2 uchar2_r = (sycl::uchar2){1,1};
  // CHECK-NEXT: auto uchar2_s = (sycl::uchar2){1,1};
  uchar2 uchar2_r = (uchar2){1,1};
  auto uchar2_s = (uchar2){1,1};
  return 0;
}

// CHECK: void func3_uchar3(sycl::uchar3 a, sycl::uchar3 b, sycl::uchar3 c) {
void func3_uchar3(uchar3 a, uchar3 b, uchar3 c) {
}
// CHECK: void func_uchar3(sycl::uchar3 a) {
void func_uchar3(uchar3 a) {
}
// CHECK: void kernel_uchar3(sycl::uchar3 *a, sycl::uchar3 *b) {
__global__ void kernel_uchar3(uchar3 *a, uchar3 *b) {
}

int main_uchar3() {
  // range default constructor does the right thing.
  // CHECK: sycl::uchar3 uchar3_a;
  uchar3 uchar3_a;
  // CHECK: sycl::uchar3 uchar3_b = sycl::uchar3(1, 2, 3);
  uchar3 uchar3_b = make_uchar3(1, 2, 3);
  // CHECK: sycl::uchar3 uchar3_c = sycl::uchar3(uchar3_b);
  uchar3 uchar3_c = uchar3(uchar3_b);
  // CHECK: sycl::uchar3 uchar3_d(uchar3_c);
  uchar3 uchar3_d(uchar3_c);
  // CHECK: func3_uchar3(uchar3_b, sycl::uchar3(uchar3_b), (sycl::uchar3)uchar3_b);
  func3_uchar3(uchar3_b, uchar3(uchar3_b), (uchar3)uchar3_b);
  // CHECK: sycl::uchar3 *uchar3_e;
  uchar3 *uchar3_e;
  // CHECK: sycl::uchar3 *uchar3_f;
  uchar3 *uchar3_f;
  // CHECK: unsigned char uchar3_g = uchar3_c.x();
  unsigned char uchar3_g = uchar3_c.x;
  // CHECK: uchar3_a.x() = uchar3_d.x();
  uchar3_a.x = uchar3_d.x;
  // CHECK: if (uchar3_b.x() == uchar3_d.x()) {}
  if (uchar3_b.x == uchar3_d.x) {}
  // CHECK: sycl::uchar3 uchar3_h[16];
  uchar3 uchar3_h[16];
  // CHECK: sycl::uchar3 uchar3_i[32];
  uchar3 uchar3_i[32];
  // CHECK: if (uchar3_h[12].x() == uchar3_i[12].x()) {}
  if (uchar3_h[12].x == uchar3_i[12].x) {}
  // CHECK: uchar3_f = (sycl::uchar3 *)uchar3_i;
  uchar3_f = (uchar3 *)uchar3_i;
  // CHECK: uchar3_a = (sycl::uchar3)uchar3_c;
  uchar3_a = (uchar3)uchar3_c;
  // CHECK: uchar3_b = sycl::uchar3(uchar3_b);
  uchar3_b = uchar3(uchar3_b);
  // CHECK: sycl::uchar3 uchar3_j, uchar3_k, uchar3_l, uchar3_m[16], *uchar3_n[32];
  uchar3 uchar3_j, uchar3_k, uchar3_l, uchar3_m[16], *uchar3_n[32];
  // CHECK: int uchar3_o = sizeof(sycl::uchar3);
  int uchar3_o = sizeof(uchar3);
  // CHECK: int unsigned char_p = sizeof(unsigned char);
  int unsigned char_p = sizeof(unsigned char);
  // CHECK: int uchar3_q = sizeof(uchar3_d);
  int uchar3_q = sizeof(uchar3_d);
  int *uchar3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uchar3 *> uchar3_e_acc_ct0(uchar3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uchar3 *> uchar3_cast_acc_ct1((sycl::uchar3 *)uchar3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_uchar3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_uchar3(uchar3_e_acc_ct0.get_raw_pointer(), uchar3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_uchar3<<<1,1>>>(uchar3_e, (uchar3 *)uchar3_cast);
  // CHECK: sycl::uchar3 uchar3_r = (sycl::uchar3){1,1,1};
  // CHECK-NEXT: auto uchar3_s = (sycl::uchar3){1,1,1};
  uchar3 uchar3_r = (uchar3){1,1,1};
  auto uchar3_s = (uchar3){1,1,1};
  return 0;
}

// CHECK: void func3_uchar4(sycl::uchar4 a, sycl::uchar4 b, sycl::uchar4 c) {
void func3_uchar4(uchar4 a, uchar4 b, uchar4 c) {
}
// CHECK: void func_uchar4(sycl::uchar4 a) {
void func_uchar4(uchar4 a) {
}
// CHECK: void kernel_uchar4(sycl::uchar4 *a, sycl::uchar4 *b) {
__global__ void kernel_uchar4(uchar4 *a, uchar4 *b) {
}

int main_uchar4() {
  // range default constructor does the right thing.
  // CHECK: sycl::uchar4 uchar4_a;
  uchar4 uchar4_a;
  // CHECK: sycl::uchar4 uchar4_b = sycl::uchar4(1, 2, 3, 4);
  uchar4 uchar4_b = make_uchar4(1, 2, 3, 4);
  // CHECK: sycl::uchar4 uchar4_c = sycl::uchar4(uchar4_b);
  uchar4 uchar4_c = uchar4(uchar4_b);
  // CHECK: sycl::uchar4 uchar4_d(uchar4_c);
  uchar4 uchar4_d(uchar4_c);
  // CHECK: func3_uchar4(uchar4_b, sycl::uchar4(uchar4_b), (sycl::uchar4)uchar4_b);
  func3_uchar4(uchar4_b, uchar4(uchar4_b), (uchar4)uchar4_b);
  // CHECK: sycl::uchar4 *uchar4_e;
  uchar4 *uchar4_e;
  // CHECK: sycl::uchar4 *uchar4_f;
  uchar4 *uchar4_f;
  // CHECK: unsigned char uchar4_g = uchar4_c.x();
  unsigned char uchar4_g = uchar4_c.x;
  // CHECK: uchar4_a.x() = uchar4_d.x();
  uchar4_a.x = uchar4_d.x;
  // CHECK: if (uchar4_b.x() == uchar4_d.x()) {}
  if (uchar4_b.x == uchar4_d.x) {}
  // CHECK: sycl::uchar4 uchar4_h[16];
  uchar4 uchar4_h[16];
  // CHECK: sycl::uchar4 uchar4_i[32];
  uchar4 uchar4_i[32];
  // CHECK: if (uchar4_h[12].x() == uchar4_i[12].x()) {}
  if (uchar4_h[12].x == uchar4_i[12].x) {}
  // CHECK: uchar4_f = (sycl::uchar4 *)uchar4_i;
  uchar4_f = (uchar4 *)uchar4_i;
  // CHECK: uchar4_a = (sycl::uchar4)uchar4_c;
  uchar4_a = (uchar4)uchar4_c;
  // CHECK: uchar4_b = sycl::uchar4(uchar4_b);
  uchar4_b = uchar4(uchar4_b);
  // CHECK: sycl::uchar4 uchar4_j, uchar4_k, uchar4_l, uchar4_m[16], *uchar4_n[32];
  uchar4 uchar4_j, uchar4_k, uchar4_l, uchar4_m[16], *uchar4_n[32];
  // CHECK: int uchar4_o = sizeof(sycl::uchar4);
  int uchar4_o = sizeof(uchar4);
  // CHECK: int unsigned char_p = sizeof(unsigned char);
  int unsigned char_p = sizeof(unsigned char);
  // CHECK: int uchar4_q = sizeof(uchar4_d);
  int uchar4_q = sizeof(uchar4_d);
  int *uchar4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uchar4 *> uchar4_e_acc_ct0(uchar4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uchar4 *> uchar4_cast_acc_ct1((sycl::uchar4 *)uchar4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_uchar4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_uchar4(uchar4_e_acc_ct0.get_raw_pointer(), uchar4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_uchar4<<<1,1>>>(uchar4_e, (uchar4 *)uchar4_cast);
  // CHECK: sycl::uchar4 uchar4_r = (sycl::uchar4){1,1,1,1};
  // CHECK-NEXT: auto uchar4_s = (sycl::uchar4){1,1,1,1};
  uchar4 uchar4_r = (uchar4){1,1,1,1};
  auto uchar4_s = (uchar4){1,1,1,1};
  return 0;
}

// CHECK: void func3_uint1(uint32_t a, uint32_t b, uint32_t c) {
void func3_uint1(uint1 a, uint1 b, uint1 c) {
}
// CHECK: void func_uint1(uint32_t a) {
void func_uint1(uint1 a) {
}
// CHECK: void kernel_uint1(uint32_t *a, uint32_t *b) {
__global__ void kernel_uint1(uint1 *a, uint1 *b) {
}

int main_uint1() {
  // range default constructor does the right thing.
  // CHECK: uint32_t uint1_a;
  uint1 uint1_a;
  // CHECK: uint32_t uint1_b = uint32_t(1);
  uint1 uint1_b = make_uint1(1);
  // CHECK: uint32_t uint1_c = uint32_t(uint1_b);
  uint1 uint1_c = uint1(uint1_b);
  // CHECK: uint32_t uint1_d(uint1_c);
  uint1 uint1_d(uint1_c);
  // CHECK: func3_uint1(uint1_b, uint32_t(uint1_b), (uint32_t)uint1_b);
  func3_uint1(uint1_b, uint1(uint1_b), (uint1)uint1_b);
  // CHECK: uint32_t *uint1_e;
  uint1 *uint1_e;
  // CHECK: uint32_t *uint1_f;
  uint1 *uint1_f;
  // CHECK: unsigned int uint1_g = uint1_c;
  unsigned int uint1_g = uint1_c.x;
  // CHECK: uint1_a = uint1_d;
  uint1_a.x = uint1_d.x;
  // CHECK: if (uint1_b == uint1_d) {}
  if (uint1_b.x == uint1_d.x) {}
  // CHECK: uint32_t uint1_h[16];
  uint1 uint1_h[16];
  // CHECK: uint32_t uint1_i[32];
  uint1 uint1_i[32];
  // CHECK: if (uint1_h[12] == uint1_i[12]) {}
  if (uint1_h[12].x == uint1_i[12].x) {}
  // CHECK: uint1_f = (uint32_t *)uint1_i;
  uint1_f = (uint1 *)uint1_i;
  // CHECK: uint1_a = (uint32_t)uint1_c;
  uint1_a = (uint1)uint1_c;
  // CHECK: uint1_b = uint32_t(uint1_b);
  uint1_b = uint1(uint1_b);
  // CHECK: uint32_t uint1_j, uint1_k, uint1_l, uint1_m[16], *uint1_n[32];
  uint1 uint1_j, uint1_k, uint1_l, uint1_m[16], *uint1_n[32];
  // CHECK: int uint1_o = sizeof(uint32_t);
  int uint1_o = sizeof(uint1);
  // CHECK: int unsigned int_p = sizeof(unsigned int);
  int unsigned int_p = sizeof(unsigned int);
  // CHECK: int uint1_q = sizeof(uint1_d);
  int uint1_q = sizeof(uint1_d);
  int *uint1_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<uint32_t *> uint1_e_acc_ct0(uint1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<uint32_t *> uint1_cast_acc_ct1((uint32_t *)uint1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_uint1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_uint1(uint1_e_acc_ct0.get_raw_pointer(), uint1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_uint1<<<1,1>>>(uint1_e, (uint1 *)uint1_cast);
  // CHECK: uint32_t uint1_r = (uint32_t){1};
  // CHECK-NEXT: auto uint1_s = (uint32_t){1};
  uint1 uint1_r = (uint1){1};
  auto uint1_s = (uint1){1};
  return 0;
}

// CHECK: void func3_uint2(sycl::uint2 a, sycl::uint2 b, sycl::uint2 c) {
void func3_uint2(uint2 a, uint2 b, uint2 c) {
}
// CHECK: void func_uint2(sycl::uint2 a) {
void func_uint2(uint2 a) {
}
// CHECK: void kernel_uint2(sycl::uint2 *a, sycl::uint2 *b) {
__global__ void kernel_uint2(uint2 *a, uint2 *b) {
}

int main_uint2() {
  // range default constructor does the right thing.
  // CHECK: sycl::uint2 uint2_a;
  uint2 uint2_a;
  // CHECK: sycl::uint2 uint2_b = sycl::uint2(1, 2);
  uint2 uint2_b = make_uint2(1, 2);
  // CHECK: sycl::uint2 uint2_c = sycl::uint2(uint2_b);
  uint2 uint2_c = uint2(uint2_b);
  // CHECK: sycl::uint2 uint2_d(uint2_c);
  uint2 uint2_d(uint2_c);
  // CHECK: func3_uint2(uint2_b, sycl::uint2(uint2_b), (sycl::uint2)uint2_b);
  func3_uint2(uint2_b, uint2(uint2_b), (uint2)uint2_b);
  // CHECK: sycl::uint2 *uint2_e;
  uint2 *uint2_e;
  // CHECK: sycl::uint2 *uint2_f;
  uint2 *uint2_f;
  // CHECK: unsigned int uint2_g = uint2_c.x();
  unsigned int uint2_g = uint2_c.x;
  // CHECK: uint2_a.x() = uint2_d.x();
  uint2_a.x = uint2_d.x;
  // CHECK: if (uint2_b.x() == uint2_d.x()) {}
  if (uint2_b.x == uint2_d.x) {}
  // CHECK: sycl::uint2 uint2_h[16];
  uint2 uint2_h[16];
  // CHECK: sycl::uint2 uint2_i[32];
  uint2 uint2_i[32];
  // CHECK: if (uint2_h[12].x() == uint2_i[12].x()) {}
  if (uint2_h[12].x == uint2_i[12].x) {}
  // CHECK: uint2_f = (sycl::uint2 *)uint2_i;
  uint2_f = (uint2 *)uint2_i;
  // CHECK: uint2_a = (sycl::uint2)uint2_c;
  uint2_a = (uint2)uint2_c;
  // CHECK: uint2_b = sycl::uint2(uint2_b);
  uint2_b = uint2(uint2_b);
  // CHECK: sycl::uint2 uint2_j, uint2_k, uint2_l, uint2_m[16], *uint2_n[32];
  uint2 uint2_j, uint2_k, uint2_l, uint2_m[16], *uint2_n[32];
  // CHECK: int uint2_o = sizeof(sycl::uint2);
  int uint2_o = sizeof(uint2);
  // CHECK: int unsigned int_p = sizeof(unsigned int);
  int unsigned int_p = sizeof(unsigned int);
  // CHECK: int uint2_q = sizeof(uint2_d);
  int uint2_q = sizeof(uint2_d);
  int *uint2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uint2 *> uint2_e_acc_ct0(uint2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uint2 *> uint2_cast_acc_ct1((sycl::uint2 *)uint2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_uint2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_uint2(uint2_e_acc_ct0.get_raw_pointer(), uint2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_uint2<<<1,1>>>(uint2_e, (uint2 *)uint2_cast);
  // CHECK: sycl::uint2 uint2_r = (sycl::uint2){1,1};
  // CHECK-NEXT: auto uint2_s = (sycl::uint2){1,1};
  uint2 uint2_r = (uint2){1,1};
  auto uint2_s = (uint2){1,1};
  return 0;
}

// CHECK: void func3_uint3(sycl::uint3 a, sycl::uint3 b, sycl::uint3 c) {
void func3_uint3(uint3 a, uint3 b, uint3 c) {
}
// CHECK: void func_uint3(sycl::uint3 a) {
void func_uint3(uint3 a) {
}
// CHECK: void kernel_uint3(sycl::uint3 *a, sycl::uint3 *b) {
__global__ void kernel_uint3(uint3 *a, uint3 *b) {
}

int main_uint3() {
  // range default constructor does the right thing.
  // CHECK: sycl::uint3 uint3_a;
  uint3 uint3_a;
  // CHECK: sycl::uint3 uint3_b = sycl::uint3(1, 2, 3);
  uint3 uint3_b = make_uint3(1, 2, 3);
  // CHECK: sycl::uint3 uint3_c = sycl::uint3(uint3_b);
  uint3 uint3_c = uint3(uint3_b);
  // CHECK: sycl::uint3 uint3_d(uint3_c);
  uint3 uint3_d(uint3_c);
  // CHECK: func3_uint3(uint3_b, sycl::uint3(uint3_b), (sycl::uint3)uint3_b);
  func3_uint3(uint3_b, uint3(uint3_b), (uint3)uint3_b);
  // CHECK: sycl::uint3 *uint3_e;
  uint3 *uint3_e;
  // CHECK: sycl::uint3 *uint3_f;
  uint3 *uint3_f;
  // CHECK: unsigned int uint3_g = uint3_c.x();
  unsigned int uint3_g = uint3_c.x;
  // CHECK: uint3_a.x() = uint3_d.x();
  uint3_a.x = uint3_d.x;
  // CHECK: if (uint3_b.x() == uint3_d.x()) {}
  if (uint3_b.x == uint3_d.x) {}
  // CHECK: sycl::uint3 uint3_h[16];
  uint3 uint3_h[16];
  // CHECK: sycl::uint3 uint3_i[32];
  uint3 uint3_i[32];
  // CHECK: if (uint3_h[12].x() == uint3_i[12].x()) {}
  if (uint3_h[12].x == uint3_i[12].x) {}
  // CHECK: uint3_f = (sycl::uint3 *)uint3_i;
  uint3_f = (uint3 *)uint3_i;
  // CHECK: uint3_a = (sycl::uint3)uint3_c;
  uint3_a = (uint3)uint3_c;
  // CHECK: uint3_b = sycl::uint3(uint3_b);
  uint3_b = uint3(uint3_b);
  // CHECK: sycl::uint3 uint3_j, uint3_k, uint3_l, uint3_m[16], *uint3_n[32];
  uint3 uint3_j, uint3_k, uint3_l, uint3_m[16], *uint3_n[32];
  // CHECK: int uint3_o = sizeof(sycl::uint3);
  int uint3_o = sizeof(uint3);
  // CHECK: int unsigned int_p = sizeof(unsigned int);
  int unsigned int_p = sizeof(unsigned int);
  // CHECK: int uint3_q = sizeof(uint3_d);
  int uint3_q = sizeof(uint3_d);
  int *uint3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uint3 *> uint3_e_acc_ct0(uint3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uint3 *> uint3_cast_acc_ct1((sycl::uint3 *)uint3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_uint3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_uint3(uint3_e_acc_ct0.get_raw_pointer(), uint3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_uint3<<<1,1>>>(uint3_e, (uint3 *)uint3_cast);
  // CHECK: sycl::uint3 uint3_r = (sycl::uint3){1,1,1};
  // CHECK-NEXT: auto uint3_s = (sycl::uint3){1,1,1};
  uint3 uint3_r = (uint3){1,1,1};
  auto uint3_s = (uint3){1,1,1};
  return 0;
}

// CHECK: void func3_uint4(sycl::uint4 a, sycl::uint4 b, sycl::uint4 c) {
void func3_uint4(uint4 a, uint4 b, uint4 c) {
}
// CHECK: void func_uint4(sycl::uint4 a) {
void func_uint4(uint4 a) {
}
// CHECK: void kernel_uint4(sycl::uint4 *a, sycl::uint4 *b) {
__global__ void kernel_uint4(uint4 *a, uint4 *b) {
}

int main_uint4() {
  // range default constructor does the right thing.
  // CHECK: sycl::uint4 uint4_a;
  uint4 uint4_a;
  // CHECK: sycl::uint4 uint4_b = sycl::uint4(1, 2, 3, 4);
  uint4 uint4_b = make_uint4(1, 2, 3, 4);
  // CHECK: sycl::uint4 uint4_c = sycl::uint4(uint4_b);
  uint4 uint4_c = uint4(uint4_b);
  // CHECK: sycl::uint4 uint4_d(uint4_c);
  uint4 uint4_d(uint4_c);
  // CHECK: func3_uint4(uint4_b, sycl::uint4(uint4_b), (sycl::uint4)uint4_b);
  func3_uint4(uint4_b, uint4(uint4_b), (uint4)uint4_b);
  // CHECK: sycl::uint4 *uint4_e;
  uint4 *uint4_e;
  // CHECK: sycl::uint4 *uint4_f;
  uint4 *uint4_f;
  // CHECK: unsigned int uint4_g = uint4_c.x();
  unsigned int uint4_g = uint4_c.x;
  // CHECK: uint4_a.x() = uint4_d.x();
  uint4_a.x = uint4_d.x;
  // CHECK: if (uint4_b.x() == uint4_d.x()) {}
  if (uint4_b.x == uint4_d.x) {}
  // CHECK: sycl::uint4 uint4_h[16];
  uint4 uint4_h[16];
  // CHECK: sycl::uint4 uint4_i[32];
  uint4 uint4_i[32];
  // CHECK: if (uint4_h[12].x() == uint4_i[12].x()) {}
  if (uint4_h[12].x == uint4_i[12].x) {}
  // CHECK: uint4_f = (sycl::uint4 *)uint4_i;
  uint4_f = (uint4 *)uint4_i;
  // CHECK: uint4_a = (sycl::uint4)uint4_c;
  uint4_a = (uint4)uint4_c;
  // CHECK: uint4_b = sycl::uint4(uint4_b);
  uint4_b = uint4(uint4_b);
  // CHECK: sycl::uint4 uint4_j, uint4_k, uint4_l, uint4_m[16], *uint4_n[32];
  uint4 uint4_j, uint4_k, uint4_l, uint4_m[16], *uint4_n[32];
  // CHECK: int uint4_o = sizeof(sycl::uint4);
  int uint4_o = sizeof(uint4);
  // CHECK: int unsigned int_p = sizeof(unsigned int);
  int unsigned int_p = sizeof(unsigned int);
  // CHECK: int uint4_q = sizeof(uint4_d);
  int uint4_q = sizeof(uint4_d);
  int *uint4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uint4 *> uint4_e_acc_ct0(uint4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::uint4 *> uint4_cast_acc_ct1((sycl::uint4 *)uint4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_uint4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_uint4(uint4_e_acc_ct0.get_raw_pointer(), uint4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_uint4<<<1,1>>>(uint4_e, (uint4 *)uint4_cast);
  // CHECK: sycl::uint4 uint4_r = (sycl::uint4){1,1,1,1};
  // CHECK-NEXT: auto uint4_s = (sycl::uint4){1,1,1,1};
  uint4 uint4_r = (uint4){1,1,1,1};
  auto uint4_s = (uint4){1,1,1,1};
  return 0;
}

// CHECK: void func3_ulong1(uint64_t a, uint64_t b, uint64_t c) {
void func3_ulong1(ulong1 a, ulong1 b, ulong1 c) {
}
// CHECK: void func_ulong1(uint64_t a) {
void func_ulong1(ulong1 a) {
}
// CHECK: void kernel_ulong1(uint64_t *a, uint64_t *b) {
__global__ void kernel_ulong1(ulong1 *a, ulong1 *b) {
}

int main_ulong1() {
  // range default constructor does the right thing.
  // CHECK: uint64_t ulong1_a;
  ulong1 ulong1_a;
  // CHECK: uint64_t ulong1_b = uint64_t(1);
  ulong1 ulong1_b = make_ulong1(1);
  // CHECK: uint64_t ulong1_c = uint64_t(ulong1_b);
  ulong1 ulong1_c = ulong1(ulong1_b);
  // CHECK: uint64_t ulong1_d(ulong1_c);
  ulong1 ulong1_d(ulong1_c);
  // CHECK: func3_ulong1(ulong1_b, uint64_t(ulong1_b), (uint64_t)ulong1_b);
  func3_ulong1(ulong1_b, ulong1(ulong1_b), (ulong1)ulong1_b);
  // CHECK: uint64_t *ulong1_e;
  ulong1 *ulong1_e;
  // CHECK: uint64_t *ulong1_f;
  ulong1 *ulong1_f;
  // CHECK: unsigned long ulong1_g = ulong1_c;
  unsigned long ulong1_g = ulong1_c.x;
  // CHECK: ulong1_a = ulong1_d;
  ulong1_a.x = ulong1_d.x;
  // CHECK: if (ulong1_b == ulong1_d) {}
  if (ulong1_b.x == ulong1_d.x) {}
  // CHECK: uint64_t ulong1_h[16];
  ulong1 ulong1_h[16];
  // CHECK: uint64_t ulong1_i[32];
  ulong1 ulong1_i[32];
  // CHECK: if (ulong1_h[12] == ulong1_i[12]) {}
  if (ulong1_h[12].x == ulong1_i[12].x) {}
  // CHECK: ulong1_f = (uint64_t *)ulong1_i;
  ulong1_f = (ulong1 *)ulong1_i;
  // CHECK: ulong1_a = (uint64_t)ulong1_c;
  ulong1_a = (ulong1)ulong1_c;
  // CHECK: ulong1_b = uint64_t(ulong1_b);
  ulong1_b = ulong1(ulong1_b);
  // CHECK: uint64_t ulong1_j, ulong1_k, ulong1_l, ulong1_m[16], *ulong1_n[32];
  ulong1 ulong1_j, ulong1_k, ulong1_l, ulong1_m[16], *ulong1_n[32];
  // CHECK: int ulong1_o = sizeof(uint64_t);
  int ulong1_o = sizeof(ulong1);
  // CHECK: int unsigned long_p = sizeof(unsigned long);
  int unsigned long_p = sizeof(unsigned long);
  // CHECK: int ulong1_q = sizeof(ulong1_d);
  int ulong1_q = sizeof(ulong1_d);
  int *ulong1_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<uint64_t *> ulong1_e_acc_ct0(ulong1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<uint64_t *> ulong1_cast_acc_ct1((uint64_t *)ulong1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ulong1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ulong1(ulong1_e_acc_ct0.get_raw_pointer(), ulong1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ulong1<<<1,1>>>(ulong1_e, (ulong1 *)ulong1_cast);
  // CHECK: uint64_t ulong1_r = (uint64_t){1};
  // CHECK-NEXT: auto ulong1_s = (uint64_t){1};
  ulong1 ulong1_r = (ulong1){1};
  auto ulong1_s = (ulong1){1};
  return 0;
}

// CHECK: void func3_ulong2(sycl::ulong2 a, sycl::ulong2 b, sycl::ulong2 c) {
void func3_ulong2(ulong2 a, ulong2 b, ulong2 c) {
}
// CHECK: void func_ulong2(sycl::ulong2 a) {
void func_ulong2(ulong2 a) {
}
// CHECK: void kernel_ulong2(sycl::ulong2 *a, sycl::ulong2 *b) {
__global__ void kernel_ulong2(ulong2 *a, ulong2 *b) {
}

int main_ulong2() {
  // range default constructor does the right thing.
  // CHECK: sycl::ulong2 ulong2_a;
  ulong2 ulong2_a;
  // CHECK: sycl::ulong2 ulong2_b = sycl::ulong2(1, 2);
  ulong2 ulong2_b = make_ulong2(1, 2);
  // CHECK: sycl::ulong2 ulong2_c = sycl::ulong2(ulong2_b);
  ulong2 ulong2_c = ulong2(ulong2_b);
  // CHECK: sycl::ulong2 ulong2_d(ulong2_c);
  ulong2 ulong2_d(ulong2_c);
  // CHECK: func3_ulong2(ulong2_b, sycl::ulong2(ulong2_b), (sycl::ulong2)ulong2_b);
  func3_ulong2(ulong2_b, ulong2(ulong2_b), (ulong2)ulong2_b);
  // CHECK: sycl::ulong2 *ulong2_e;
  ulong2 *ulong2_e;
  // CHECK: sycl::ulong2 *ulong2_f;
  ulong2 *ulong2_f;
  // CHECK: unsigned long ulong2_g = ulong2_c.x();
  unsigned long ulong2_g = ulong2_c.x;
  // CHECK: ulong2_a.x() = ulong2_d.x();
  ulong2_a.x = ulong2_d.x;
  // CHECK: if (ulong2_b.x() == ulong2_d.x()) {}
  if (ulong2_b.x == ulong2_d.x) {}
  // CHECK: sycl::ulong2 ulong2_h[16];
  ulong2 ulong2_h[16];
  // CHECK: sycl::ulong2 ulong2_i[32];
  ulong2 ulong2_i[32];
  // CHECK: if (ulong2_h[12].x() == ulong2_i[12].x()) {}
  if (ulong2_h[12].x == ulong2_i[12].x) {}
  // CHECK: ulong2_f = (sycl::ulong2 *)ulong2_i;
  ulong2_f = (ulong2 *)ulong2_i;
  // CHECK: ulong2_a = (sycl::ulong2)ulong2_c;
  ulong2_a = (ulong2)ulong2_c;
  // CHECK: ulong2_b = sycl::ulong2(ulong2_b);
  ulong2_b = ulong2(ulong2_b);
  // CHECK: sycl::ulong2 ulong2_j, ulong2_k, ulong2_l, ulong2_m[16], *ulong2_n[32];
  ulong2 ulong2_j, ulong2_k, ulong2_l, ulong2_m[16], *ulong2_n[32];
  // CHECK: int ulong2_o = sizeof(sycl::ulong2);
  int ulong2_o = sizeof(ulong2);
  // CHECK: int unsigned long_p = sizeof(unsigned long);
  int unsigned long_p = sizeof(unsigned long);
  // CHECK: int ulong2_q = sizeof(ulong2_d);
  int ulong2_q = sizeof(ulong2_d);
  int *ulong2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulong2 *> ulong2_e_acc_ct0(ulong2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulong2 *> ulong2_cast_acc_ct1((sycl::ulong2 *)ulong2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ulong2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ulong2(ulong2_e_acc_ct0.get_raw_pointer(), ulong2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ulong2<<<1,1>>>(ulong2_e, (ulong2 *)ulong2_cast);
  // CHECK: sycl::ulong2 ulong2_r = (sycl::ulong2){1,1};
  // CHECK-NEXT: auto ulong2_s = (sycl::ulong2){1,1};
  ulong2 ulong2_r = (ulong2){1,1};
  auto ulong2_s = (ulong2){1,1};
  return 0;
}

// CHECK: void func3_ulong3(sycl::ulong3 a, sycl::ulong3 b, sycl::ulong3 c) {
void func3_ulong3(ulong3 a, ulong3 b, ulong3 c) {
}
// CHECK: void func_ulong3(sycl::ulong3 a) {
void func_ulong3(ulong3 a) {
}
// CHECK: void kernel_ulong3(sycl::ulong3 *a, sycl::ulong3 *b) {
__global__ void kernel_ulong3(ulong3 *a, ulong3 *b) {
}

int main_ulong3() {
  // range default constructor does the right thing.
  // CHECK: sycl::ulong3 ulong3_a;
  ulong3 ulong3_a;
  // CHECK: sycl::ulong3 ulong3_b = sycl::ulong3(1, 2, 3);
  ulong3 ulong3_b = make_ulong3(1, 2, 3);
  // CHECK: sycl::ulong3 ulong3_c = sycl::ulong3(ulong3_b);
  ulong3 ulong3_c = ulong3(ulong3_b);
  // CHECK: sycl::ulong3 ulong3_d(ulong3_c);
  ulong3 ulong3_d(ulong3_c);
  // CHECK: func3_ulong3(ulong3_b, sycl::ulong3(ulong3_b), (sycl::ulong3)ulong3_b);
  func3_ulong3(ulong3_b, ulong3(ulong3_b), (ulong3)ulong3_b);
  // CHECK: sycl::ulong3 *ulong3_e;
  ulong3 *ulong3_e;
  // CHECK: sycl::ulong3 *ulong3_f;
  ulong3 *ulong3_f;
  // CHECK: unsigned long ulong3_g = ulong3_c.x();
  unsigned long ulong3_g = ulong3_c.x;
  // CHECK: ulong3_a.x() = ulong3_d.x();
  ulong3_a.x = ulong3_d.x;
  // CHECK: if (ulong3_b.x() == ulong3_d.x()) {}
  if (ulong3_b.x == ulong3_d.x) {}
  // CHECK: sycl::ulong3 ulong3_h[16];
  ulong3 ulong3_h[16];
  // CHECK: sycl::ulong3 ulong3_i[32];
  ulong3 ulong3_i[32];
  // CHECK: if (ulong3_h[12].x() == ulong3_i[12].x()) {}
  if (ulong3_h[12].x == ulong3_i[12].x) {}
  // CHECK: ulong3_f = (sycl::ulong3 *)ulong3_i;
  ulong3_f = (ulong3 *)ulong3_i;
  // CHECK: ulong3_a = (sycl::ulong3)ulong3_c;
  ulong3_a = (ulong3)ulong3_c;
  // CHECK: ulong3_b = sycl::ulong3(ulong3_b);
  ulong3_b = ulong3(ulong3_b);
  // CHECK: sycl::ulong3 ulong3_j, ulong3_k, ulong3_l, ulong3_m[16], *ulong3_n[32];
  ulong3 ulong3_j, ulong3_k, ulong3_l, ulong3_m[16], *ulong3_n[32];
  // CHECK: int ulong3_o = sizeof(sycl::ulong3);
  int ulong3_o = sizeof(ulong3);
  // CHECK: int unsigned long_p = sizeof(unsigned long);
  int unsigned long_p = sizeof(unsigned long);
  // CHECK: int ulong3_q = sizeof(ulong3_d);
  int ulong3_q = sizeof(ulong3_d);
  int *ulong3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulong3 *> ulong3_e_acc_ct0(ulong3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulong3 *> ulong3_cast_acc_ct1((sycl::ulong3 *)ulong3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ulong3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ulong3(ulong3_e_acc_ct0.get_raw_pointer(), ulong3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ulong3<<<1,1>>>(ulong3_e, (ulong3 *)ulong3_cast);
  // CHECK: sycl::ulong3 ulong3_r = (sycl::ulong3){1,1,1};
  // CHECK-NEXT: auto ulong3_s = (sycl::ulong3){1,1,1};
  ulong3 ulong3_r = (ulong3){1,1,1};
  auto ulong3_s = (ulong3){1,1,1};
  return 0;
}

// CHECK: void func3_ulong4(sycl::ulong4 a, sycl::ulong4 b, sycl::ulong4 c) {
void func3_ulong4(ulong4 a, ulong4 b, ulong4 c) {
}
// CHECK: void func_ulong4(sycl::ulong4 a) {
void func_ulong4(ulong4 a) {
}
// CHECK: void kernel_ulong4(sycl::ulong4 *a, sycl::ulong4 *b) {
__global__ void kernel_ulong4(ulong4 *a, ulong4 *b) {
}

int main_ulong4() {
  // range default constructor does the right thing.
  // CHECK: sycl::ulong4 ulong4_a;
  ulong4 ulong4_a;
  // CHECK: sycl::ulong4 ulong4_b = sycl::ulong4(1, 2, 3, 4);
  ulong4 ulong4_b = make_ulong4(1, 2, 3, 4);
  // CHECK: sycl::ulong4 ulong4_c = sycl::ulong4(ulong4_b);
  ulong4 ulong4_c = ulong4(ulong4_b);
  // CHECK: sycl::ulong4 ulong4_d(ulong4_c);
  ulong4 ulong4_d(ulong4_c);
  // CHECK: func3_ulong4(ulong4_b, sycl::ulong4(ulong4_b), (sycl::ulong4)ulong4_b);
  func3_ulong4(ulong4_b, ulong4(ulong4_b), (ulong4)ulong4_b);
  // CHECK: sycl::ulong4 *ulong4_e;
  ulong4 *ulong4_e;
  // CHECK: sycl::ulong4 *ulong4_f;
  ulong4 *ulong4_f;
  // CHECK: unsigned long ulong4_g = ulong4_c.x();
  unsigned long ulong4_g = ulong4_c.x;
  // CHECK: ulong4_a.x() = ulong4_d.x();
  ulong4_a.x = ulong4_d.x;
  // CHECK: if (ulong4_b.x() == ulong4_d.x()) {}
  if (ulong4_b.x == ulong4_d.x) {}
  // CHECK: sycl::ulong4 ulong4_h[16];
  ulong4 ulong4_h[16];
  // CHECK: sycl::ulong4 ulong4_i[32];
  ulong4 ulong4_i[32];
  // CHECK: if (ulong4_h[12].x() == ulong4_i[12].x()) {}
  if (ulong4_h[12].x == ulong4_i[12].x) {}
  // CHECK: ulong4_f = (sycl::ulong4 *)ulong4_i;
  ulong4_f = (ulong4 *)ulong4_i;
  // CHECK: ulong4_a = (sycl::ulong4)ulong4_c;
  ulong4_a = (ulong4)ulong4_c;
  // CHECK: ulong4_b = sycl::ulong4(ulong4_b);
  ulong4_b = ulong4(ulong4_b);
  // CHECK: sycl::ulong4 ulong4_j, ulong4_k, ulong4_l, ulong4_m[16], *ulong4_n[32];
  ulong4 ulong4_j, ulong4_k, ulong4_l, ulong4_m[16], *ulong4_n[32];
  // CHECK: int ulong4_o = sizeof(sycl::ulong4);
  int ulong4_o = sizeof(ulong4);
  // CHECK: int unsigned long_p = sizeof(unsigned long);
  int unsigned long_p = sizeof(unsigned long);
  // CHECK: int ulong4_q = sizeof(ulong4_d);
  int ulong4_q = sizeof(ulong4_d);
  int *ulong4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulong4 *> ulong4_e_acc_ct0(ulong4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulong4 *> ulong4_cast_acc_ct1((sycl::ulong4 *)ulong4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ulong4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ulong4(ulong4_e_acc_ct0.get_raw_pointer(), ulong4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ulong4<<<1,1>>>(ulong4_e, (ulong4 *)ulong4_cast);
  // CHECK: sycl::ulong4 ulong4_r = (sycl::ulong4){1,1,1,1};
  // CHECK-NEXT: auto ulong4_s = (sycl::ulong4){1,1,1,1};
  ulong4 ulong4_r = (ulong4){1,1,1,1};
  auto ulong4_s = (ulong4){1,1,1,1};
  return 0;
}

// CHECK: void func3_ulonglong1(uint64_t a, uint64_t b, uint64_t c) {
void func3_ulonglong1(ulonglong1 a, ulonglong1 b, ulonglong1 c) {
}
// CHECK: void func_ulonglong1(uint64_t a) {
void func_ulonglong1(ulonglong1 a) {
}
// CHECK: void kernel_ulonglong1(uint64_t *a, uint64_t *b) {
__global__ void kernel_ulonglong1(ulonglong1 *a, ulonglong1 *b) {
}

int main_ulonglong1() {
  // range default constructor does the right thing.
  // CHECK: uint64_t ulonglong1_a;
  ulonglong1 ulonglong1_a;
  // CHECK: uint64_t ulonglong1_b = uint64_t(1);
  ulonglong1 ulonglong1_b = make_ulonglong1(1);
  // CHECK: uint64_t ulonglong1_c = uint64_t(ulonglong1_b);
  ulonglong1 ulonglong1_c = ulonglong1(ulonglong1_b);
  // CHECK: uint64_t ulonglong1_d(ulonglong1_c);
  ulonglong1 ulonglong1_d(ulonglong1_c);
  // CHECK: func3_ulonglong1(ulonglong1_b, uint64_t(ulonglong1_b), (uint64_t)ulonglong1_b);
  func3_ulonglong1(ulonglong1_b, ulonglong1(ulonglong1_b), (ulonglong1)ulonglong1_b);
  // CHECK: uint64_t *ulonglong1_e;
  ulonglong1 *ulonglong1_e;
  // CHECK: uint64_t *ulonglong1_f;
  ulonglong1 *ulonglong1_f;
  // CHECK: unsigned long long ulonglong1_g = ulonglong1_c;
  unsigned long long ulonglong1_g = ulonglong1_c.x;
  // CHECK: ulonglong1_a = ulonglong1_d;
  ulonglong1_a.x = ulonglong1_d.x;
  // CHECK: if (ulonglong1_b == ulonglong1_d) {}
  if (ulonglong1_b.x == ulonglong1_d.x) {}
  // CHECK: uint64_t ulonglong1_h[16];
  ulonglong1 ulonglong1_h[16];
  // CHECK: uint64_t ulonglong1_i[32];
  ulonglong1 ulonglong1_i[32];
  // CHECK: if (ulonglong1_h[12] == ulonglong1_i[12]) {}
  if (ulonglong1_h[12].x == ulonglong1_i[12].x) {}
  // CHECK: ulonglong1_f = (uint64_t *)ulonglong1_i;
  ulonglong1_f = (ulonglong1 *)ulonglong1_i;
  // CHECK: ulonglong1_a = (uint64_t)ulonglong1_c;
  ulonglong1_a = (ulonglong1)ulonglong1_c;
  // CHECK: ulonglong1_b = uint64_t(ulonglong1_b);
  ulonglong1_b = ulonglong1(ulonglong1_b);
  // CHECK: uint64_t ulonglong1_j, ulonglong1_k, ulonglong1_l, ulonglong1_m[16], *ulonglong1_n[32];
  ulonglong1 ulonglong1_j, ulonglong1_k, ulonglong1_l, ulonglong1_m[16], *ulonglong1_n[32];
  // CHECK: int ulonglong1_o = sizeof(uint64_t);
  int ulonglong1_o = sizeof(ulonglong1);
  // CHECK: int unsigned long long_p = sizeof(unsigned long long);
  int unsigned long long_p = sizeof(unsigned long long);
  // CHECK: int ulonglong1_q = sizeof(ulonglong1_d);
  int ulonglong1_q = sizeof(ulonglong1_d);
  int *ulonglong1_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<uint64_t *> ulonglong1_e_acc_ct0(ulonglong1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<uint64_t *> ulonglong1_cast_acc_ct1((uint64_t *)ulonglong1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ulonglong1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ulonglong1(ulonglong1_e_acc_ct0.get_raw_pointer(), ulonglong1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ulonglong1<<<1,1>>>(ulonglong1_e, (ulonglong1 *)ulonglong1_cast);
  // CHECK: uint64_t ulonglong1_r = (uint64_t){1};
  // CHECK-NEXT: auto ulonglong1_s = (uint64_t){1};
  ulonglong1 ulonglong1_r = (ulonglong1){1};
  auto ulonglong1_s = (ulonglong1){1};
  return 0;
}

// CHECK: void func3_ulonglong2(sycl::ulonglong2 a, sycl::ulonglong2 b, sycl::ulonglong2 c) {
void func3_ulonglong2(ulonglong2 a, ulonglong2 b, ulonglong2 c) {
}
// CHECK: void func_ulonglong2(sycl::ulonglong2 a) {
void func_ulonglong2(ulonglong2 a) {
}
// CHECK: void kernel_ulonglong2(sycl::ulonglong2 *a, sycl::ulonglong2 *b) {
__global__ void kernel_ulonglong2(ulonglong2 *a, ulonglong2 *b) {
}

int main_ulonglong2() {
  // range default constructor does the right thing.
  // CHECK: sycl::ulonglong2 ulonglong2_a;
  ulonglong2 ulonglong2_a;
  // CHECK: sycl::ulonglong2 ulonglong2_b = sycl::ulonglong2(1, 2);
  ulonglong2 ulonglong2_b = make_ulonglong2(1, 2);
  // CHECK: sycl::ulonglong2 ulonglong2_c = sycl::ulonglong2(ulonglong2_b);
  ulonglong2 ulonglong2_c = ulonglong2(ulonglong2_b);
  // CHECK: sycl::ulonglong2 ulonglong2_d(ulonglong2_c);
  ulonglong2 ulonglong2_d(ulonglong2_c);
  // CHECK: func3_ulonglong2(ulonglong2_b, sycl::ulonglong2(ulonglong2_b), (sycl::ulonglong2)ulonglong2_b);
  func3_ulonglong2(ulonglong2_b, ulonglong2(ulonglong2_b), (ulonglong2)ulonglong2_b);
  // CHECK: sycl::ulonglong2 *ulonglong2_e;
  ulonglong2 *ulonglong2_e;
  // CHECK: sycl::ulonglong2 *ulonglong2_f;
  ulonglong2 *ulonglong2_f;
  // CHECK: unsigned long long ulonglong2_g = ulonglong2_c.x();
  unsigned long long ulonglong2_g = ulonglong2_c.x;
  // CHECK: ulonglong2_a.x() = ulonglong2_d.x();
  ulonglong2_a.x = ulonglong2_d.x;
  // CHECK: if (ulonglong2_b.x() == ulonglong2_d.x()) {}
  if (ulonglong2_b.x == ulonglong2_d.x) {}
  // CHECK: sycl::ulonglong2 ulonglong2_h[16];
  ulonglong2 ulonglong2_h[16];
  // CHECK: sycl::ulonglong2 ulonglong2_i[32];
  ulonglong2 ulonglong2_i[32];
  // CHECK: if (ulonglong2_h[12].x() == ulonglong2_i[12].x()) {}
  if (ulonglong2_h[12].x == ulonglong2_i[12].x) {}
  // CHECK: ulonglong2_f = (sycl::ulonglong2 *)ulonglong2_i;
  ulonglong2_f = (ulonglong2 *)ulonglong2_i;
  // CHECK: ulonglong2_a = (sycl::ulonglong2)ulonglong2_c;
  ulonglong2_a = (ulonglong2)ulonglong2_c;
  // CHECK: ulonglong2_b = sycl::ulonglong2(ulonglong2_b);
  ulonglong2_b = ulonglong2(ulonglong2_b);
  // CHECK: sycl::ulonglong2 ulonglong2_j, ulonglong2_k, ulonglong2_l, ulonglong2_m[16], *ulonglong2_n[32];
  ulonglong2 ulonglong2_j, ulonglong2_k, ulonglong2_l, ulonglong2_m[16], *ulonglong2_n[32];
  // CHECK: int ulonglong2_o = sizeof(sycl::ulonglong2);
  int ulonglong2_o = sizeof(ulonglong2);
  // CHECK: int unsigned long long_p = sizeof(unsigned long long);
  int unsigned long long_p = sizeof(unsigned long long);
  // CHECK: int ulonglong2_q = sizeof(ulonglong2_d);
  int ulonglong2_q = sizeof(ulonglong2_d);
  int *ulonglong2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulonglong2 *> ulonglong2_e_acc_ct0(ulonglong2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulonglong2 *> ulonglong2_cast_acc_ct1((sycl::ulonglong2 *)ulonglong2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ulonglong2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ulonglong2(ulonglong2_e_acc_ct0.get_raw_pointer(), ulonglong2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ulonglong2<<<1,1>>>(ulonglong2_e, (ulonglong2 *)ulonglong2_cast);
  // CHECK: sycl::ulonglong2 ulonglong2_r = (sycl::ulonglong2){1,1};
  // CHECK-NEXT: auto ulonglong2_s = (sycl::ulonglong2){1,1};
  ulonglong2 ulonglong2_r = (ulonglong2){1,1};
  auto ulonglong2_s = (ulonglong2){1,1};
  return 0;
}

// CHECK: void func3_ulonglong3(sycl::ulonglong3 a, sycl::ulonglong3 b, sycl::ulonglong3 c) {
void func3_ulonglong3(ulonglong3 a, ulonglong3 b, ulonglong3 c) {
}
// CHECK: void func_ulonglong3(sycl::ulonglong3 a) {
void func_ulonglong3(ulonglong3 a) {
}
// CHECK: void kernel_ulonglong3(sycl::ulonglong3 *a, sycl::ulonglong3 *b) {
__global__ void kernel_ulonglong3(ulonglong3 *a, ulonglong3 *b) {
}

int main_ulonglong3() {
  // range default constructor does the right thing.
  // CHECK: sycl::ulonglong3 ulonglong3_a;
  ulonglong3 ulonglong3_a;
  // CHECK: sycl::ulonglong3 ulonglong3_b = sycl::ulonglong3(1, 2, 3);
  ulonglong3 ulonglong3_b = make_ulonglong3(1, 2, 3);
  // CHECK: sycl::ulonglong3 ulonglong3_c = sycl::ulonglong3(ulonglong3_b);
  ulonglong3 ulonglong3_c = ulonglong3(ulonglong3_b);
  // CHECK: sycl::ulonglong3 ulonglong3_d(ulonglong3_c);
  ulonglong3 ulonglong3_d(ulonglong3_c);
  // CHECK: func3_ulonglong3(ulonglong3_b, sycl::ulonglong3(ulonglong3_b), (sycl::ulonglong3)ulonglong3_b);
  func3_ulonglong3(ulonglong3_b, ulonglong3(ulonglong3_b), (ulonglong3)ulonglong3_b);
  // CHECK: sycl::ulonglong3 *ulonglong3_e;
  ulonglong3 *ulonglong3_e;
  // CHECK: sycl::ulonglong3 *ulonglong3_f;
  ulonglong3 *ulonglong3_f;
  // CHECK: unsigned long long ulonglong3_g = ulonglong3_c.x();
  unsigned long long ulonglong3_g = ulonglong3_c.x;
  // CHECK: ulonglong3_a.x() = ulonglong3_d.x();
  ulonglong3_a.x = ulonglong3_d.x;
  // CHECK: if (ulonglong3_b.x() == ulonglong3_d.x()) {}
  if (ulonglong3_b.x == ulonglong3_d.x) {}
  // CHECK: sycl::ulonglong3 ulonglong3_h[16];
  ulonglong3 ulonglong3_h[16];
  // CHECK: sycl::ulonglong3 ulonglong3_i[32];
  ulonglong3 ulonglong3_i[32];
  // CHECK: if (ulonglong3_h[12].x() == ulonglong3_i[12].x()) {}
  if (ulonglong3_h[12].x == ulonglong3_i[12].x) {}
  // CHECK: ulonglong3_f = (sycl::ulonglong3 *)ulonglong3_i;
  ulonglong3_f = (ulonglong3 *)ulonglong3_i;
  // CHECK: ulonglong3_a = (sycl::ulonglong3)ulonglong3_c;
  ulonglong3_a = (ulonglong3)ulonglong3_c;
  // CHECK: ulonglong3_b = sycl::ulonglong3(ulonglong3_b);
  ulonglong3_b = ulonglong3(ulonglong3_b);
  // CHECK: sycl::ulonglong3 ulonglong3_j, ulonglong3_k, ulonglong3_l, ulonglong3_m[16], *ulonglong3_n[32];
  ulonglong3 ulonglong3_j, ulonglong3_k, ulonglong3_l, ulonglong3_m[16], *ulonglong3_n[32];
  // CHECK: int ulonglong3_o = sizeof(sycl::ulonglong3);
  int ulonglong3_o = sizeof(ulonglong3);
  // CHECK: int unsigned long long_p = sizeof(unsigned long long);
  int unsigned long long_p = sizeof(unsigned long long);
  // CHECK: int ulonglong3_q = sizeof(ulonglong3_d);
  int ulonglong3_q = sizeof(ulonglong3_d);
  int *ulonglong3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulonglong3 *> ulonglong3_e_acc_ct0(ulonglong3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulonglong3 *> ulonglong3_cast_acc_ct1((sycl::ulonglong3 *)ulonglong3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ulonglong3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ulonglong3(ulonglong3_e_acc_ct0.get_raw_pointer(), ulonglong3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ulonglong3<<<1,1>>>(ulonglong3_e, (ulonglong3 *)ulonglong3_cast);
  // CHECK: sycl::ulonglong3 ulonglong3_r = (sycl::ulonglong3){1,1,1};
  // CHECK-NEXT: auto ulonglong3_s = (sycl::ulonglong3){1,1,1};
  ulonglong3 ulonglong3_r = (ulonglong3){1,1,1};
  auto ulonglong3_s = (ulonglong3){1,1,1};
  return 0;
}

// CHECK: void func3_ulonglong4(sycl::ulonglong4 a, sycl::ulonglong4 b, sycl::ulonglong4 c) {
void func3_ulonglong4(ulonglong4 a, ulonglong4 b, ulonglong4 c) {
}
// CHECK: void func_ulonglong4(sycl::ulonglong4 a) {
void func_ulonglong4(ulonglong4 a) {
}
// CHECK: void kernel_ulonglong4(sycl::ulonglong4 *a, sycl::ulonglong4 *b) {
__global__ void kernel_ulonglong4(ulonglong4 *a, ulonglong4 *b) {
}

int main_ulonglong4() {
  // range default constructor does the right thing.
  // CHECK: sycl::ulonglong4 ulonglong4_a;
  ulonglong4 ulonglong4_a;
  // CHECK: sycl::ulonglong4 ulonglong4_b = sycl::ulonglong4(1, 2, 3, 4);
  ulonglong4 ulonglong4_b = make_ulonglong4(1, 2, 3, 4);
  // CHECK: sycl::ulonglong4 ulonglong4_c = sycl::ulonglong4(ulonglong4_b);
  ulonglong4 ulonglong4_c = ulonglong4(ulonglong4_b);
  // CHECK: sycl::ulonglong4 ulonglong4_d(ulonglong4_c);
  ulonglong4 ulonglong4_d(ulonglong4_c);
  // CHECK: func3_ulonglong4(ulonglong4_b, sycl::ulonglong4(ulonglong4_b), (sycl::ulonglong4)ulonglong4_b);
  func3_ulonglong4(ulonglong4_b, ulonglong4(ulonglong4_b), (ulonglong4)ulonglong4_b);
  // CHECK: sycl::ulonglong4 *ulonglong4_e;
  ulonglong4 *ulonglong4_e;
  // CHECK: sycl::ulonglong4 *ulonglong4_f;
  ulonglong4 *ulonglong4_f;
  // CHECK: unsigned long long ulonglong4_g = ulonglong4_c.x();
  unsigned long long ulonglong4_g = ulonglong4_c.x;
  // CHECK: ulonglong4_a.x() = ulonglong4_d.x();
  ulonglong4_a.x = ulonglong4_d.x;
  // CHECK: if (ulonglong4_b.x() == ulonglong4_d.x()) {}
  if (ulonglong4_b.x == ulonglong4_d.x) {}
  // CHECK: sycl::ulonglong4 ulonglong4_h[16];
  ulonglong4 ulonglong4_h[16];
  // CHECK: sycl::ulonglong4 ulonglong4_i[32];
  ulonglong4 ulonglong4_i[32];
  // CHECK: if (ulonglong4_h[12].x() == ulonglong4_i[12].x()) {}
  if (ulonglong4_h[12].x == ulonglong4_i[12].x) {}
  // CHECK: ulonglong4_f = (sycl::ulonglong4 *)ulonglong4_i;
  ulonglong4_f = (ulonglong4 *)ulonglong4_i;
  // CHECK: ulonglong4_a = (sycl::ulonglong4)ulonglong4_c;
  ulonglong4_a = (ulonglong4)ulonglong4_c;
  // CHECK: ulonglong4_b = sycl::ulonglong4(ulonglong4_b);
  ulonglong4_b = ulonglong4(ulonglong4_b);
  // CHECK: sycl::ulonglong4 ulonglong4_j, ulonglong4_k, ulonglong4_l, ulonglong4_m[16], *ulonglong4_n[32];
  ulonglong4 ulonglong4_j, ulonglong4_k, ulonglong4_l, ulonglong4_m[16], *ulonglong4_n[32];
  // CHECK: int ulonglong4_o = sizeof(sycl::ulonglong4);
  int ulonglong4_o = sizeof(ulonglong4);
  // CHECK: int unsigned long long_p = sizeof(unsigned long long);
  int unsigned long long_p = sizeof(unsigned long long);
  // CHECK: int ulonglong4_q = sizeof(ulonglong4_d);
  int ulonglong4_q = sizeof(ulonglong4_d);
  int *ulonglong4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulonglong4 *> ulonglong4_e_acc_ct0(ulonglong4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ulonglong4 *> ulonglong4_cast_acc_ct1((sycl::ulonglong4 *)ulonglong4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ulonglong4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ulonglong4(ulonglong4_e_acc_ct0.get_raw_pointer(), ulonglong4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ulonglong4<<<1,1>>>(ulonglong4_e, (ulonglong4 *)ulonglong4_cast);
  // CHECK: sycl::ulonglong4 ulonglong4_r = (sycl::ulonglong4){1,1,1,1};
  // CHECK-NEXT: auto ulonglong4_s = (sycl::ulonglong4){1,1,1,1};
  ulonglong4 ulonglong4_r = (ulonglong4){1,1,1,1};
  auto ulonglong4_s = (ulonglong4){1,1,1,1};
  return 0;
}

// CHECK: void func3_ushort1(uint16_t a, uint16_t b, uint16_t c) {
void func3_ushort1(ushort1 a, ushort1 b, ushort1 c) {
}
// CHECK: void func_ushort1(uint16_t a) {
void func_ushort1(ushort1 a) {
}
// CHECK: void kernel_ushort1(uint16_t *a, uint16_t *b) {
__global__ void kernel_ushort1(ushort1 *a, ushort1 *b) {
}

int main_ushort1() {
  // range default constructor does the right thing.
  // CHECK: uint16_t ushort1_a;
  ushort1 ushort1_a;
  // CHECK: uint16_t ushort1_b = uint16_t(1);
  ushort1 ushort1_b = make_ushort1(1);
  // CHECK: uint16_t ushort1_c = uint16_t(ushort1_b);
  ushort1 ushort1_c = ushort1(ushort1_b);
  // CHECK: uint16_t ushort1_d(ushort1_c);
  ushort1 ushort1_d(ushort1_c);
  // CHECK: func3_ushort1(ushort1_b, uint16_t(ushort1_b), (uint16_t)ushort1_b);
  func3_ushort1(ushort1_b, ushort1(ushort1_b), (ushort1)ushort1_b);
  // CHECK: uint16_t *ushort1_e;
  ushort1 *ushort1_e;
  // CHECK: uint16_t *ushort1_f;
  ushort1 *ushort1_f;
  // CHECK: unsigned short ushort1_g = ushort1_c;
  unsigned short ushort1_g = ushort1_c.x;
  // CHECK: ushort1_a = ushort1_d;
  ushort1_a.x = ushort1_d.x;
  // CHECK: if (ushort1_b == ushort1_d) {}
  if (ushort1_b.x == ushort1_d.x) {}
  // CHECK: uint16_t ushort1_h[16];
  ushort1 ushort1_h[16];
  // CHECK: uint16_t ushort1_i[32];
  ushort1 ushort1_i[32];
  // CHECK: if (ushort1_h[12] == ushort1_i[12]) {}
  if (ushort1_h[12].x == ushort1_i[12].x) {}
  // CHECK: ushort1_f = (uint16_t *)ushort1_i;
  ushort1_f = (ushort1 *)ushort1_i;
  // CHECK: ushort1_a = (uint16_t)ushort1_c;
  ushort1_a = (ushort1)ushort1_c;
  // CHECK: ushort1_b = uint16_t(ushort1_b);
  ushort1_b = ushort1(ushort1_b);
  // CHECK: uint16_t ushort1_j, ushort1_k, ushort1_l, ushort1_m[16], *ushort1_n[32];
  ushort1 ushort1_j, ushort1_k, ushort1_l, ushort1_m[16], *ushort1_n[32];
  // CHECK: int ushort1_o = sizeof(uint16_t);
  int ushort1_o = sizeof(ushort1);
  // CHECK: int unsigned short_p = sizeof(unsigned short);
  int unsigned short_p = sizeof(unsigned short);
  // CHECK: int ushort1_q = sizeof(ushort1_d);
  int ushort1_q = sizeof(ushort1_d);
  int *ushort1_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<uint16_t *> ushort1_e_acc_ct0(ushort1_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<uint16_t *> ushort1_cast_acc_ct1((uint16_t *)ushort1_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ushort1_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ushort1(ushort1_e_acc_ct0.get_raw_pointer(), ushort1_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ushort1<<<1,1>>>(ushort1_e, (ushort1 *)ushort1_cast);
  // CHECK: uint16_t ushort1_r = (uint16_t){1};
  // CHECK-NEXT: auto ushort1_s = (uint16_t){1};
  ushort1 ushort1_r = (ushort1){1};
  auto ushort1_s = (ushort1){1};
  return 0;
}

// CHECK: void func3_ushort2(sycl::ushort2 a, sycl::ushort2 b, sycl::ushort2 c) {
void func3_ushort2(ushort2 a, ushort2 b, ushort2 c) {
}
// CHECK: void func_ushort2(sycl::ushort2 a) {
void func_ushort2(ushort2 a) {
}
// CHECK: void kernel_ushort2(sycl::ushort2 *a, sycl::ushort2 *b) {
__global__ void kernel_ushort2(ushort2 *a, ushort2 *b) {
}

int main_ushort2() {
  // range default constructor does the right thing.
  // CHECK: sycl::ushort2 ushort2_a;
  ushort2 ushort2_a;
  // CHECK: sycl::ushort2 ushort2_b = sycl::ushort2(1, 2);
  ushort2 ushort2_b = make_ushort2(1, 2);
  // CHECK: sycl::ushort2 ushort2_c = sycl::ushort2(ushort2_b);
  ushort2 ushort2_c = ushort2(ushort2_b);
  // CHECK: sycl::ushort2 ushort2_d(ushort2_c);
  ushort2 ushort2_d(ushort2_c);
  // CHECK: func3_ushort2(ushort2_b, sycl::ushort2(ushort2_b), (sycl::ushort2)ushort2_b);
  func3_ushort2(ushort2_b, ushort2(ushort2_b), (ushort2)ushort2_b);
  // CHECK: sycl::ushort2 *ushort2_e;
  ushort2 *ushort2_e;
  // CHECK: sycl::ushort2 *ushort2_f;
  ushort2 *ushort2_f;
  // CHECK: unsigned short ushort2_g = ushort2_c.x();
  unsigned short ushort2_g = ushort2_c.x;
  // CHECK: ushort2_a.x() = ushort2_d.x();
  ushort2_a.x = ushort2_d.x;
  // CHECK: if (ushort2_b.x() == ushort2_d.x()) {}
  if (ushort2_b.x == ushort2_d.x) {}
  // CHECK: sycl::ushort2 ushort2_h[16];
  ushort2 ushort2_h[16];
  // CHECK: sycl::ushort2 ushort2_i[32];
  ushort2 ushort2_i[32];
  // CHECK: if (ushort2_h[12].x() == ushort2_i[12].x()) {}
  if (ushort2_h[12].x == ushort2_i[12].x) {}
  // CHECK: ushort2_f = (sycl::ushort2 *)ushort2_i;
  ushort2_f = (ushort2 *)ushort2_i;
  // CHECK: ushort2_a = (sycl::ushort2)ushort2_c;
  ushort2_a = (ushort2)ushort2_c;
  // CHECK: ushort2_b = sycl::ushort2(ushort2_b);
  ushort2_b = ushort2(ushort2_b);
  // CHECK: sycl::ushort2 ushort2_j, ushort2_k, ushort2_l, ushort2_m[16], *ushort2_n[32];
  ushort2 ushort2_j, ushort2_k, ushort2_l, ushort2_m[16], *ushort2_n[32];
  // CHECK: int ushort2_o = sizeof(sycl::ushort2);
  int ushort2_o = sizeof(ushort2);
  // CHECK: int unsigned short_p = sizeof(unsigned short);
  int unsigned short_p = sizeof(unsigned short);
  // CHECK: int ushort2_q = sizeof(ushort2_d);
  int ushort2_q = sizeof(ushort2_d);
  int *ushort2_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ushort2 *> ushort2_e_acc_ct0(ushort2_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ushort2 *> ushort2_cast_acc_ct1((sycl::ushort2 *)ushort2_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ushort2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ushort2(ushort2_e_acc_ct0.get_raw_pointer(), ushort2_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ushort2<<<1,1>>>(ushort2_e, (ushort2 *)ushort2_cast);
  // CHECK: sycl::ushort2 ushort2_r = (sycl::ushort2){1,1};
  // CHECK-NEXT: auto ushort2_s = (sycl::ushort2){1,1};
  ushort2 ushort2_r = (ushort2){1,1};
  auto ushort2_s = (ushort2){1,1};
  return 0;
}

// CHECK: void func3_ushort3(sycl::ushort3 a, sycl::ushort3 b, sycl::ushort3 c) {
void func3_ushort3(ushort3 a, ushort3 b, ushort3 c) {
}
// CHECK: void func_ushort3(sycl::ushort3 a) {
void func_ushort3(ushort3 a) {
}
// CHECK: void kernel_ushort3(sycl::ushort3 *a, sycl::ushort3 *b) {
__global__ void kernel_ushort3(ushort3 *a, ushort3 *b) {
}

int main_ushort3() {
  // range default constructor does the right thing.
  // CHECK: sycl::ushort3 ushort3_a;
  ushort3 ushort3_a;
  // CHECK: sycl::ushort3 ushort3_b = sycl::ushort3(1, 2, 3);
  ushort3 ushort3_b = make_ushort3(1, 2, 3);
  // CHECK: sycl::ushort3 ushort3_c = sycl::ushort3(ushort3_b);
  ushort3 ushort3_c = ushort3(ushort3_b);
  // CHECK: sycl::ushort3 ushort3_d(ushort3_c);
  ushort3 ushort3_d(ushort3_c);
  // CHECK: func3_ushort3(ushort3_b, sycl::ushort3(ushort3_b), (sycl::ushort3)ushort3_b);
  func3_ushort3(ushort3_b, ushort3(ushort3_b), (ushort3)ushort3_b);
  // CHECK: sycl::ushort3 *ushort3_e;
  ushort3 *ushort3_e;
  // CHECK: sycl::ushort3 *ushort3_f;
  ushort3 *ushort3_f;
  // CHECK: unsigned short ushort3_g = ushort3_c.x();
  unsigned short ushort3_g = ushort3_c.x;
  // CHECK: ushort3_a.x() = ushort3_d.x();
  ushort3_a.x = ushort3_d.x;
  // CHECK: if (ushort3_b.x() == ushort3_d.x()) {}
  if (ushort3_b.x == ushort3_d.x) {}
  // CHECK: sycl::ushort3 ushort3_h[16];
  ushort3 ushort3_h[16];
  // CHECK: sycl::ushort3 ushort3_i[32];
  ushort3 ushort3_i[32];
  // CHECK: if (ushort3_h[12].x() == ushort3_i[12].x()) {}
  if (ushort3_h[12].x == ushort3_i[12].x) {}
  // CHECK: ushort3_f = (sycl::ushort3 *)ushort3_i;
  ushort3_f = (ushort3 *)ushort3_i;
  // CHECK: ushort3_a = (sycl::ushort3)ushort3_c;
  ushort3_a = (ushort3)ushort3_c;
  // CHECK: ushort3_b = sycl::ushort3(ushort3_b);
  ushort3_b = ushort3(ushort3_b);
  // CHECK: sycl::ushort3 ushort3_j, ushort3_k, ushort3_l, ushort3_m[16], *ushort3_n[32];
  ushort3 ushort3_j, ushort3_k, ushort3_l, ushort3_m[16], *ushort3_n[32];
  // CHECK: int ushort3_o = sizeof(sycl::ushort3);
  int ushort3_o = sizeof(ushort3);
  // CHECK: int unsigned short_p = sizeof(unsigned short);
  int unsigned short_p = sizeof(unsigned short);
  // CHECK: int ushort3_q = sizeof(ushort3_d);
  int ushort3_q = sizeof(ushort3_d);
  int *ushort3_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ushort3 *> ushort3_e_acc_ct0(ushort3_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ushort3 *> ushort3_cast_acc_ct1((sycl::ushort3 *)ushort3_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ushort3_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ushort3(ushort3_e_acc_ct0.get_raw_pointer(), ushort3_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ushort3<<<1,1>>>(ushort3_e, (ushort3 *)ushort3_cast);
  // CHECK: sycl::ushort3 ushort3_r = (sycl::ushort3){1,1,1};
  // CHECK-NEXT: auto ushort3_s = (sycl::ushort3){1,1,1};
  ushort3 ushort3_r = (ushort3){1,1,1};
  auto ushort3_s = (ushort3){1,1,1};
  return 0;
}

// CHECK: void func3_ushort4(sycl::ushort4 a, sycl::ushort4 b, sycl::ushort4 c) {
void func3_ushort4(ushort4 a, ushort4 b, ushort4 c) {
}
// CHECK: void func_ushort4(sycl::ushort4 a) {
void func_ushort4(ushort4 a) {
}
// CHECK: void kernel_ushort4(sycl::ushort4 *a, sycl::ushort4 *b) {
__global__ void kernel_ushort4(ushort4 *a, ushort4 *b) {
}

int main_ushort4() {
  // range default constructor does the right thing.
  // CHECK: sycl::ushort4 ushort4_a;
  ushort4 ushort4_a;
  // CHECK: sycl::ushort4 ushort4_b = sycl::ushort4(1, 2, 3, 4);
  ushort4 ushort4_b = make_ushort4(1, 2, 3, 4);
  // CHECK: sycl::ushort4 ushort4_c = sycl::ushort4(ushort4_b);
  ushort4 ushort4_c = ushort4(ushort4_b);
  // CHECK: sycl::ushort4 ushort4_d(ushort4_c);
  ushort4 ushort4_d(ushort4_c);
  // CHECK: func3_ushort4(ushort4_b, sycl::ushort4(ushort4_b), (sycl::ushort4)ushort4_b);
  func3_ushort4(ushort4_b, ushort4(ushort4_b), (ushort4)ushort4_b);
  // CHECK: sycl::ushort4 *ushort4_e;
  ushort4 *ushort4_e;
  // CHECK: sycl::ushort4 *ushort4_f;
  ushort4 *ushort4_f;
  // CHECK: unsigned short ushort4_g = ushort4_c.x();
  unsigned short ushort4_g = ushort4_c.x;
  // CHECK: ushort4_a.x() = ushort4_d.x();
  ushort4_a.x = ushort4_d.x;
  // CHECK: if (ushort4_b.x() == ushort4_d.x()) {}
  if (ushort4_b.x == ushort4_d.x) {}
  // CHECK: sycl::ushort4 ushort4_h[16];
  ushort4 ushort4_h[16];
  // CHECK: sycl::ushort4 ushort4_i[32];
  ushort4 ushort4_i[32];
  // CHECK: if (ushort4_h[12].x() == ushort4_i[12].x()) {}
  if (ushort4_h[12].x == ushort4_i[12].x) {}
  // CHECK: ushort4_f = (sycl::ushort4 *)ushort4_i;
  ushort4_f = (ushort4 *)ushort4_i;
  // CHECK: ushort4_a = (sycl::ushort4)ushort4_c;
  ushort4_a = (ushort4)ushort4_c;
  // CHECK: ushort4_b = sycl::ushort4(ushort4_b);
  ushort4_b = ushort4(ushort4_b);
  // CHECK: sycl::ushort4 ushort4_j, ushort4_k, ushort4_l, ushort4_m[16], *ushort4_n[32];
  ushort4 ushort4_j, ushort4_k, ushort4_l, ushort4_m[16], *ushort4_n[32];
  // CHECK: int ushort4_o = sizeof(sycl::ushort4);
  int ushort4_o = sizeof(ushort4);
  // CHECK: int unsigned short_p = sizeof(unsigned short);
  int unsigned short_p = sizeof(unsigned short);
  // CHECK: int ushort4_q = sizeof(ushort4_d);
  int ushort4_q = sizeof(ushort4_d);
  int *ushort4_cast;
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ushort4 *> ushort4_e_acc_ct0(ushort4_e, cgh);
  // CHECK-NEXT:     dpct::access_wrapper<sycl::ushort4 *> ushort4_cast_acc_ct1((sycl::ushort4 *)ushort4_cast, cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for<dpct_kernel_name<class kernel_ushort4_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         kernel_ushort4(ushort4_e_acc_ct0.get_raw_pointer(), ushort4_cast_acc_ct1.get_raw_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  kernel_ushort4<<<1,1>>>(ushort4_e, (ushort4 *)ushort4_cast);
  // CHECK: sycl::ushort4 ushort4_r = (sycl::ushort4){1,1,1,1};
  // CHECK-NEXT: auto ushort4_s = (sycl::ushort4){1,1,1,1};
  ushort4 ushort4_r = (ushort4){1,1,1,1};
  auto ushort4_s = (ushort4){1,1,1,1};
  return 0;
}

// CHECK: void  foo_type() {
// CHECK-NEXT: const unsigned int * base;
// CHECK-NEXT: sycl::uint2 p_1;
// CHECK-NEXT: p_1 = ((const sycl::uint2 *)base)[0];
// CHECK-NEXT: sycl::int2 p_2 = ((const sycl::int2 *)base)[0];
// CHECK-NEXT: }
__global__ void  foo_type() {
const unsigned int * base;
uint2 p_1;
p_1 = ((const uint2 *)base)[0];
int2 p_2 = ((const int2 *)base)[0];
}

union struct_union {
  // CHECK: sycl::float2 data{};
  float2 data;
};

int foo() {
  struct_union temp;
  // CHECK: int x = temp.data.x() + 2;
  int x = temp.data.x + 2;
}

// CHECK: std::vector<const char> const foo_char1(std::vector<const char> a);
// CHECK-NEXT: std::vector<sycl::char2> foo_char2(std::vector<sycl::char2> a);
// CHECK-NEXT: std::vector<sycl::char3> foo_char3(std::vector<sycl::char3> a);
// CHECK-NEXT: std::vector<sycl::char4> foo_char4(std::vector<sycl::char4> a);
std::vector<const char1> const foo_char1(std::vector<const char1> a);
std::vector<char2> foo_char2(std::vector<char2> a);
std::vector<char3> foo_char3(std::vector<char3> a);
std::vector<char4> foo_char4(std::vector<char4> a);

// CHECK: std::vector<const uint8_t> const foo_uchar1(std::vector<const uint8_t> a);
// CHECK-NEXT: std::vector<sycl::uchar2> foo_uchar2(std::vector<sycl::uchar2> a);
// CHECK-NEXT: std::vector<sycl::uchar3> foo_uchar3(std::vector<sycl::uchar3> a);
// CHECK-NEXT: std::vector<sycl::uchar4> foo_uchar4(std::vector<sycl::uchar4> a);
std::vector<const uchar1> const foo_uchar1(std::vector<const uchar1> a);
std::vector<uchar2> foo_uchar2(std::vector<uchar2> a);
std::vector<uchar3> foo_uchar3(std::vector<uchar3> a);
std::vector<uchar4> foo_uchar4(std::vector<uchar4> a);

// CHECK: std::vector<const short> const foo_short1(std::vector<const short> a);
// CHECK-NEXT: std::vector<sycl::short2> foo_short2(std::vector<sycl::short2> a);
// CHECK-NEXT: std::vector<sycl::short3> foo_short3(std::vector<sycl::short3> a);
// CHECK-NEXT: std::vector<sycl::short4> foo_short4(std::vector<sycl::short4> a);
std::vector<const short1> const foo_short1(std::vector<const short1> a);
std::vector<short2> foo_short2(std::vector<short2> a);
std::vector<short3> foo_short3(std::vector<short3> a);
std::vector<short4> foo_short4(std::vector<short4> a);

// CHECK: std::vector<const uint16_t> const foo_ushort1(std::vector<const uint16_t> a);
// CHECK-NEXT: std::vector<sycl::ushort2> foo_ushort2(std::vector<sycl::ushort2> a);
// CHECK-NEXT: std::vector<sycl::ushort3> foo_ushort3(std::vector<sycl::ushort3> a);
// CHECK-NEXT: std::vector<sycl::ushort4> foo_ushort4(std::vector<sycl::ushort4> a);
std::vector<const ushort1> const foo_ushort1(std::vector<const ushort1> a);
std::vector<ushort2> foo_ushort2(std::vector<ushort2> a);
std::vector<ushort3> foo_ushort3(std::vector<ushort3> a);
std::vector<ushort4> foo_ushort4(std::vector<ushort4> a);

// CHECK: std::vector<const int> const foo_int1(std::vector<const int> a);
// CHECK-NEXT: std::vector<sycl::int2> foo_int2(std::vector<sycl::int2> a);
// CHECK-NEXT: std::vector<sycl::int3> foo_int3(std::vector<sycl::int3> a);
// CHECK-NEXT: std::vector<sycl::int4> foo_int4(std::vector<sycl::int4> a);
std::vector<const int1> const foo_int1(std::vector<const int1> a);
std::vector<int2> foo_int2(std::vector<int2> a);
std::vector<int3> foo_int3(std::vector<int3> a);
std::vector<int4> foo_int4(std::vector<int4> a);

// CHECK: std::vector<const uint32_t> const foo_uint1(std::vector<const uint32_t> a);
// CHECK-NEXT: std::vector<sycl::uint2> foo_uint2(std::vector<sycl::uint2> a);
// CHECK-NEXT: std::vector<sycl::uint3> foo_uint3(std::vector<sycl::uint3> a);
// CHECK-NEXT: std::vector<sycl::uint4> foo_uint4(std::vector<sycl::uint4> a);
std::vector<const uint1> const foo_uint1(std::vector<const uint1> a);
std::vector<uint2> foo_uint2(std::vector<uint2> a);
std::vector<uint3> foo_uint3(std::vector<uint3> a);
std::vector<uint4> foo_uint4(std::vector<uint4> a);

// CHECK: std::vector<const long> const foo_long1(std::vector<const long> a);
// CHECK-NEXT: std::vector<sycl::long2> foo_long2(std::vector<sycl::long2> a);
// CHECK-NEXT: std::vector<sycl::long3> foo_long3(std::vector<sycl::long3> a);
// CHECK-NEXT: std::vector<sycl::long4> foo_long4(std::vector<sycl::long4> a);
std::vector<const long1> const foo_long1(std::vector<const long1> a);
std::vector<long2> foo_long2(std::vector<long2> a);
std::vector<long3> foo_long3(std::vector<long3> a);
std::vector<long4> foo_long4(std::vector<long4> a);

// CHECK: std::vector<const uint64_t> const foo_ulong1(std::vector<const uint64_t> a);
// CHECK-NEXT: std::vector<sycl::ulong2> foo_ulong2(std::vector<sycl::ulong2> a);
// CHECK-NEXT: std::vector<sycl::ulong3> foo_ulong3(std::vector<sycl::ulong3> a);
// CHECK-NEXT: std::vector<sycl::ulong4> foo_ulong4(std::vector<sycl::ulong4> a);
std::vector<const ulong1> const foo_ulong1(std::vector<const ulong1> a);
std::vector<ulong2> foo_ulong2(std::vector<ulong2> a);
std::vector<ulong3> foo_ulong3(std::vector<ulong3> a);
std::vector<ulong4> foo_ulong4(std::vector<ulong4> a);

// CHECK: std::vector<const float> const foo_float1(std::vector<const float> a);
// CHECK-NEXT: std::vector<sycl::float2> foo_float2(std::vector<sycl::float2> a);
// CHECK-NEXT: std::vector<sycl::float3> foo_float3(std::vector<sycl::float3> a);
// CHECK-NEXT: std::vector<sycl::float4> foo_float4(std::vector<sycl::float4> a);
std::vector<const float1> const foo_float1(std::vector<const float1> a);
std::vector<float2> foo_float2(std::vector<float2> a);
std::vector<float3> foo_float3(std::vector<float3> a);
std::vector<float4> foo_float4(std::vector<float4> a);

// CHECK: std::vector<const int64_t> const foo_longlong1(std::vector<const int64_t> a);
// CHECK-NEXT: std::vector<sycl::longlong2> foo_longlong2(std::vector<sycl::longlong2> a);
// CHECK-NEXT: std::vector<sycl::longlong3> foo_longlong3(std::vector<sycl::longlong3> a);
// CHECK-NEXT: std::vector<sycl::longlong4> foo_longlong4(std::vector<sycl::longlong4> a);
std::vector<const longlong1> const foo_longlong1(std::vector<const longlong1> a);
std::vector<longlong2> foo_longlong2(std::vector<longlong2> a);
std::vector<longlong3> foo_longlong3(std::vector<longlong3> a);
std::vector<longlong4> foo_longlong4(std::vector<longlong4> a);

// CHECK: std::vector<const uint64_t> const foo_ulonglong1(std::vector<const uint64_t> a);
// CHECK-NEXT: std::vector<sycl::ulonglong2> foo_ulonglong2(std::vector<sycl::ulonglong2> a);
// CHECK-NEXT: std::vector<sycl::ulonglong3> foo_ulonglong3(std::vector<sycl::ulonglong3> a);
// CHECK-NEXT: std::vector<sycl::ulonglong4> foo_ulonglong4(std::vector<sycl::ulonglong4> a);
std::vector<const ulonglong1> const foo_ulonglong1(std::vector<const ulonglong1> a);
std::vector<ulonglong2> foo_ulonglong2(std::vector<ulonglong2> a);
std::vector<ulonglong3> foo_ulonglong3(std::vector<ulonglong3> a);
std::vector<ulonglong4> foo_ulonglong4(std::vector<ulonglong4> a);

// CHECK: std::vector<const double> const foo_double1(std::vector<const double> a);
// CHECK-NEXT: std::vector<sycl::double2> foo_double2(std::vector<sycl::double2> a);
// CHECK-NEXT: std::vector<sycl::double3> foo_double3(std::vector<sycl::double3> a);
// CHECK-NEXT: std::vector<sycl::double4> foo_double4(std::vector<sycl::double4> a);
std::vector<const double1> const foo_double1(std::vector<const double1> a);
std::vector<double2> foo_double2(std::vector<double2> a);
std::vector<double3> foo_double3(std::vector<double3> a);
std::vector<double4> foo_double4(std::vector<double4> a);

void bar(){
  int a;
  // CHECK: std::vector<sycl::char2> *bar_char2 = (std::vector<sycl::char2>*) a;
  // CHECK-NEXT: std::vector<sycl::uchar2> *bar_uchar2 = (std::vector<sycl::uchar2>*) a;
  // CHECK-NEXT: std::vector<sycl::short2> *bar_short2 = (std::vector<sycl::short2>*) a;
  // CHECK-NEXT: std::vector<sycl::ushort2> *bar_ushort2 = (std::vector<sycl::ushort2>*) a;
  // CHECK-NEXT: std::vector<sycl::int2> *bar_int2 = (std::vector<sycl::int2>*) a;
  // CHECK-NEXT: std::vector<sycl::uint2> *bar_uint2 = (std::vector<sycl::uint2>*) a;
  // CHECK-NEXT: std::vector<sycl::long2> *bar_long2 = (std::vector<sycl::long2>*) a;
  // CHECK-NEXT: std::vector<sycl::ulong2> *bar_ulong2 = (std::vector<sycl::ulong2>*) a;
  // CHECK-NEXT: std::vector<sycl::float2> *bar_float2 = (std::vector<sycl::float2>*) a;
  // CHECK-NEXT: std::vector<sycl::longlong2> *bar_longlong2 = (std::vector<sycl::longlong2>*) a;
  // CHECK-NEXT: std::vector<sycl::ulonglong2> *bar_ulonglong2 = (std::vector<sycl::ulonglong2>*) a;
  // CHECK-NEXT: std::vector<sycl::double2> *bar_double2 = (std::vector<sycl::double2>*) a;
  std::vector<char2> *bar_char2 = (std::vector<char2>*) a;
  std::vector<uchar2> *bar_uchar2 = (std::vector<uchar2>*) a;
  std::vector<short2> *bar_short2 = (std::vector<short2>*) a;
  std::vector<ushort2> *bar_ushort2 = (std::vector<ushort2>*) a;
  std::vector<int2> *bar_int2 = (std::vector<int2>*) a;
  std::vector<uint2> *bar_uint2 = (std::vector<uint2>*) a;
  std::vector<long2> *bar_long2 = (std::vector<long2>*) a;
  std::vector<ulong2> *bar_ulong2 = (std::vector<ulong2>*) a;
  std::vector<float2> *bar_float2 = (std::vector<float2>*) a;
  std::vector<longlong2> *bar_longlong2 = (std::vector<longlong2>*) a;
  std::vector<ulonglong2> *bar_ulonglong2 = (std::vector<ulonglong2>*) a;
  std::vector<double2> *bar_double2 = (std::vector<double2>*) a;


// CHECK: #define vchar2_ptr std::vector<sycl::char2>*
// CHECK-NEXT: #define vuchar2_ptr std::vector<sycl::uchar2>*
// CHECK-NEXT: #define vshort2_ptr std::vector<sycl::short2>*
// CHECK-NEXT: #define vushort2_ptr std::vector<sycl::ushort2>*
// CHECK-NEXT: #define vint2_ptr std::vector<sycl::int2>*
// CHECK-NEXT: #define vuint2_ptr std::vector<sycl::uint2>*
// CHECK-NEXT: #define vlong2_ptr std::vector<sycl::long2>*
// CHECK-NEXT: #define vulong2_ptr std::vector<sycl::ulong2>*
// CHECK-NEXT: #define vfloat2_ptr std::vector<sycl::float2>*
// CHECK-NEXT: #define vlonglong2_ptr std::vector<sycl::longlong2>*
// CHECK-NEXT: #define vulonglong2_ptr std::vector<sycl::ulonglong2>*
// CHECK-NEXT: #define vdouble2_ptr std::vector<sycl::double2>*
// CHECK-NEXT:   bar_char2 = (vchar2_ptr) a;
// CHECK-NEXT:    bar_uchar2 = (vuchar2_ptr) a;
// CHECK-NEXT:   bar_short2 = (vshort2_ptr) a;
// CHECK-NEXT:   bar_ushort2 = (vushort2_ptr) a;
// CHECK-NEXT:   bar_int2 = (vint2_ptr) a;
// CHECK-NEXT:   bar_uint2 = (vuint2_ptr) a;
// CHECK-NEXT:   bar_long2 = (vlong2_ptr) a;
// CHECK-NEXT:   bar_ulong2 = (vulong2_ptr) a;
// CHECK-NEXT:   bar_float2 = (vfloat2_ptr) a;
// CHECK-NEXT:   bar_longlong2 = (vlonglong2_ptr) a;
// CHECK-NEXT:   bar_ulonglong2 = (vulonglong2_ptr) a;
// CHECK-NEXT:   bar_double2 = (vdouble2_ptr) a;
#define vchar2_ptr std::vector<char2>*
#define vuchar2_ptr std::vector<uchar2>*
#define vshort2_ptr std::vector<short2>*
#define vushort2_ptr std::vector<ushort2>*
#define vint2_ptr std::vector<int2>*
#define vuint2_ptr std::vector<uint2>*
#define vlong2_ptr std::vector<long2>*
#define vulong2_ptr std::vector<ulong2>*
#define vfloat2_ptr std::vector<float2>*
#define vlonglong2_ptr std::vector<longlong2>*
#define vulonglong2_ptr std::vector<ulonglong2>*
#define vdouble2_ptr std::vector<double2>*
  bar_char2 = (vchar2_ptr) a;
  bar_uchar2 = (vuchar2_ptr) a;
  bar_short2 = (vshort2_ptr) a;
  bar_ushort2 = (vushort2_ptr) a;
  bar_int2 = (vint2_ptr) a;
  bar_uint2 = (vuint2_ptr) a;
  bar_long2 = (vlong2_ptr) a;
  bar_ulong2 = (vulong2_ptr) a;
  bar_float2 = (vfloat2_ptr) a;
  bar_longlong2 = (vlonglong2_ptr) a;
  bar_ulonglong2 = (vulonglong2_ptr) a;
  bar_double2 = (vdouble2_ptr) a;
}


using VT = double2;
typedef double2 VT2;

void test() {
  //CHECK: VT d2;
  //CHECK-NEXT: if (std::abs(d2.x()) >= 0.01) {}
  VT d2;
  if (std::abs(d2.x) >= 0.01) {}
}

void test2() {
  //CHECK: VT2 d2;
  //CHECK-NEXT: if (std::abs(d2.x()) >= 0.01) {}
  VT2 d2;
  if (std::abs(d2.x) >= 0.01) {}
}

