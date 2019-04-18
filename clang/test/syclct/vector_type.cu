// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/vector_type.sycl.cpp --match-full-lines %s

// CHECK: void func3_char1(char a, char b, char c) try {
void func3_char1(char1 a, char1 b, char1 c) {
}
// CHECK: void func_char1(char a) try {
void func_char1(char1 a) {
}
// CHECK: void kernel_char1(char *a) {
__global__ void kernel_char1(char1 *a) {
}

int main_char1() {
  // range default constructor does the right thing.
  // CHECK: char a;
  char1 a;
  // CHECK: char b = char(1);
  char1 b = make_char1(1);
  // CHECK: char c = char(b);
  char1 c = char1(b);
  // CHECK: char d(c);
  char1 d(c);
  // CHECK: func3_char1(b, char(b), (char)b);
  func3_char1(b, char1(b), (char1)b);
  // CHECK: char *e;
  char1 *e;
  // CHECK: char *f;
  char1 *f;
  // CHECK: signed char g = static_cast<signed char>(c);
  signed char g = c.x;
  // CHECK: a = static_cast<signed char>(d);
  a.x = d.x;
  // CHECK: if (static_cast<signed char>(b) == static_cast<signed char>(d)) {}
  if (b.x == d.x) {}
  // CHECK: char h[16];
  char1 h[16];
  // CHECK: char i[32];
  char1 i[32];
  // CHECK: if (static_cast<signed char>(h[12]) == static_cast<signed char>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (char *)i;
  f = (char1 *)i;
  // CHECK: a = (char)c;
  a = (char1)c;
  // CHECK: b = char(c);
  b = char1(c);
  // CHECK: char j, k, l, m[16], *n[32];
  char1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(char);
  int o = sizeof(char1);
  // CHECK: int p = sizeof(signed char);
  int p = sizeof(signed char);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_char1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           char *e = (char*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_char1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_char1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_char2(cl::sycl::char2 a, cl::sycl::char2 b, cl::sycl::char2 c) try {
void func3_char2(char2 a, char2 b, char2 c) {
}
// CHECK: void func_char2(cl::sycl::char2 a) try {
void func_char2(char2 a) {
}
// CHECK: void kernel_char2(cl::sycl::char2 *a) {
__global__ void kernel_char2(char2 *a) {
}

int main_char2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::char2 a;
  char2 a;
  // CHECK: cl::sycl::char2 b = cl::sycl::char2(1, 2);
  char2 b = make_char2(1, 2);
  // CHECK: cl::sycl::char2 c = cl::sycl::char2(b);
  char2 c = char2(b);
  // CHECK: cl::sycl::char2 d(c);
  char2 d(c);
  // CHECK: func3_char2(b, cl::sycl::char2(b), (cl::sycl::char2)b);
  func3_char2(b, char2(b), (char2)b);
  // CHECK: cl::sycl::char2 *e;
  char2 *e;
  // CHECK: cl::sycl::char2 *f;
  char2 *f;
  // CHECK: signed char g = static_cast<signed char>(c.x());
  signed char g = c.x;
  // CHECK: a.x() = static_cast<signed char>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<signed char>(b.x()) == static_cast<signed char>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::char2 h[16];
  char2 h[16];
  // CHECK: cl::sycl::char2 i[32];
  char2 i[32];
  // CHECK: if (static_cast<signed char>(h[12].x()) == static_cast<signed char>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::char2 *)i;
  f = (char2 *)i;
  // CHECK: a = (cl::sycl::char2)c;
  a = (char2)c;
  // CHECK: b = cl::sycl::char2(c);
  b = char2(c);
  // CHECK: cl::sycl::char2 j, k, l, m[16], *n[32];
  char2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::char2);
  int o = sizeof(char2);
  // CHECK: int p = sizeof(signed char);
  int p = sizeof(signed char);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_char2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::char2 *e = (cl::sycl::char2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_char2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_char2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_char3(cl::sycl::char3 a, cl::sycl::char3 b, cl::sycl::char3 c) try {
void func3_char3(char3 a, char3 b, char3 c) {
}
// CHECK: void func_char3(cl::sycl::char3 a) try {
void func_char3(char3 a) {
}
// CHECK: void kernel_char3(cl::sycl::char3 *a) {
__global__ void kernel_char3(char3 *a) {
}

int main_char3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::char3 a;
  char3 a;
  // CHECK: cl::sycl::char3 b = cl::sycl::char3(1, 2, 3);
  char3 b = make_char3(1, 2, 3);
  // CHECK: cl::sycl::char3 c = cl::sycl::char3(b);
  char3 c = char3(b);
  // CHECK: cl::sycl::char3 d(c);
  char3 d(c);
  // CHECK: func3_char3(b, cl::sycl::char3(b), (cl::sycl::char3)b);
  func3_char3(b, char3(b), (char3)b);
  // CHECK: cl::sycl::char3 *e;
  char3 *e;
  // CHECK: cl::sycl::char3 *f;
  char3 *f;
  // CHECK: signed char g = static_cast<signed char>(c.x());
  signed char g = c.x;
  // CHECK: a.x() = static_cast<signed char>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<signed char>(b.x()) == static_cast<signed char>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::char3 h[16];
  char3 h[16];
  // CHECK: cl::sycl::char3 i[32];
  char3 i[32];
  // CHECK: if (static_cast<signed char>(h[12].x()) == static_cast<signed char>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::char3 *)i;
  f = (char3 *)i;
  // CHECK: a = (cl::sycl::char3)c;
  a = (char3)c;
  // CHECK: b = cl::sycl::char3(c);
  b = char3(c);
  // CHECK: cl::sycl::char3 j, k, l, m[16], *n[32];
  char3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::char3);
  int o = sizeof(char3);
  // CHECK: int p = sizeof(signed char);
  int p = sizeof(signed char);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_char3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::char3 *e = (cl::sycl::char3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_char3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_char3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_char4(cl::sycl::char4 a, cl::sycl::char4 b, cl::sycl::char4 c) try {
void func3_char4(char4 a, char4 b, char4 c) {
}
// CHECK: void func_char4(cl::sycl::char4 a) try {
void func_char4(char4 a) {
}
// CHECK: void kernel_char4(cl::sycl::char4 *a) {
__global__ void kernel_char4(char4 *a) {
}

int main_char4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::char4 a;
  char4 a;
  // CHECK: cl::sycl::char4 b = cl::sycl::char4(1, 2, 3, 4);
  char4 b = make_char4(1, 2, 3, 4);
  // CHECK: cl::sycl::char4 c = cl::sycl::char4(b);
  char4 c = char4(b);
  // CHECK: cl::sycl::char4 d(c);
  char4 d(c);
  // CHECK: func3_char4(b, cl::sycl::char4(b), (cl::sycl::char4)b);
  func3_char4(b, char4(b), (char4)b);
  // CHECK: cl::sycl::char4 *e;
  char4 *e;
  // CHECK: cl::sycl::char4 *f;
  char4 *f;
  // CHECK: signed char g = static_cast<signed char>(c.x());
  signed char g = c.x;
  // CHECK: a.x() = static_cast<signed char>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<signed char>(b.x()) == static_cast<signed char>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::char4 h[16];
  char4 h[16];
  // CHECK: cl::sycl::char4 i[32];
  char4 i[32];
  // CHECK: if (static_cast<signed char>(h[12].x()) == static_cast<signed char>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::char4 *)i;
  f = (char4 *)i;
  // CHECK: a = (cl::sycl::char4)c;
  a = (char4)c;
  // CHECK: b = cl::sycl::char4(c);
  b = char4(c);
  // CHECK: cl::sycl::char4 j, k, l, m[16], *n[32];
  char4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::char4);
  int o = sizeof(char4);
  // CHECK: int p = sizeof(signed char);
  int p = sizeof(signed char);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_char4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::char4 *e = (cl::sycl::char4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_char4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_char4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_double1(double a, double b, double c) try {
void func3_double1(double1 a, double1 b, double1 c) {
}
// CHECK: void func_double1(double a) try {
void func_double1(double1 a) {
}
// CHECK: void kernel_double1(double *a) {
__global__ void kernel_double1(double1 *a) {
}

int main_double1() {
  // range default constructor does the right thing.
  // CHECK: double a;
  double1 a;
  // CHECK: double b = double(1);
  double1 b = make_double1(1);
  // CHECK: double c = double(b);
  double1 c = double1(b);
  // CHECK: double d(c);
  double1 d(c);
  // CHECK: func3_double1(b, double(b), (double)b);
  func3_double1(b, double1(b), (double1)b);
  // CHECK: double *e;
  double1 *e;
  // CHECK: double *f;
  double1 *f;
  // CHECK: double g = static_cast<double>(c);
  double g = c.x;
  // CHECK: a = static_cast<double>(d);
  a.x = d.x;
  // CHECK: if (static_cast<double>(b) == static_cast<double>(d)) {}
  if (b.x == d.x) {}
  // CHECK: double h[16];
  double1 h[16];
  // CHECK: double i[32];
  double1 i[32];
  // CHECK: if (static_cast<double>(h[12]) == static_cast<double>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (double *)i;
  f = (double1 *)i;
  // CHECK: a = (double)c;
  a = (double1)c;
  // CHECK: b = double(c);
  b = double1(c);
  // CHECK: double j, k, l, m[16], *n[32];
  double1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(double);
  int o = sizeof(double1);
  // CHECK: int p = sizeof(double);
  int p = sizeof(double);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_double1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           double *e = (double*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_double1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_double1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_double2(cl::sycl::double2 a, cl::sycl::double2 b, cl::sycl::double2 c) try {
void func3_double2(double2 a, double2 b, double2 c) {
}
// CHECK: void func_double2(cl::sycl::double2 a) try {
void func_double2(double2 a) {
}
// CHECK: void kernel_double2(cl::sycl::double2 *a) {
__global__ void kernel_double2(double2 *a) {
}

int main_double2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::double2 a;
  double2 a;
  // CHECK: cl::sycl::double2 b = cl::sycl::double2(1, 2);
  double2 b = make_double2(1, 2);
  // CHECK: cl::sycl::double2 c = cl::sycl::double2(b);
  double2 c = double2(b);
  // CHECK: cl::sycl::double2 d(c);
  double2 d(c);
  // CHECK: func3_double2(b, cl::sycl::double2(b), (cl::sycl::double2)b);
  func3_double2(b, double2(b), (double2)b);
  // CHECK: cl::sycl::double2 *e;
  double2 *e;
  // CHECK: cl::sycl::double2 *f;
  double2 *f;
  // CHECK: double g = static_cast<double>(c.x());
  double g = c.x;
  // CHECK: a.x() = static_cast<double>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<double>(b.x()) == static_cast<double>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::double2 h[16];
  double2 h[16];
  // CHECK: cl::sycl::double2 i[32];
  double2 i[32];
  // CHECK: if (static_cast<double>(h[12].x()) == static_cast<double>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::double2 *)i;
  f = (double2 *)i;
  // CHECK: a = (cl::sycl::double2)c;
  a = (double2)c;
  // CHECK: b = cl::sycl::double2(c);
  b = double2(c);
  // CHECK: cl::sycl::double2 j, k, l, m[16], *n[32];
  double2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::double2);
  int o = sizeof(double2);
  // CHECK: int p = sizeof(double);
  int p = sizeof(double);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_double2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::double2 *e = (cl::sycl::double2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_double2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_double2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_double3(cl::sycl::double3 a, cl::sycl::double3 b, cl::sycl::double3 c) try {
void func3_double3(double3 a, double3 b, double3 c) {
}
// CHECK: void func_double3(cl::sycl::double3 a) try {
void func_double3(double3 a) {
}
// CHECK: void kernel_double3(cl::sycl::double3 *a) {
__global__ void kernel_double3(double3 *a) {
}

int main_double3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::double3 a;
  double3 a;
  // CHECK: cl::sycl::double3 b = cl::sycl::double3(1, 2, 3);
  double3 b = make_double3(1, 2, 3);
  // CHECK: cl::sycl::double3 c = cl::sycl::double3(b);
  double3 c = double3(b);
  // CHECK: cl::sycl::double3 d(c);
  double3 d(c);
  // CHECK: func3_double3(b, cl::sycl::double3(b), (cl::sycl::double3)b);
  func3_double3(b, double3(b), (double3)b);
  // CHECK: cl::sycl::double3 *e;
  double3 *e;
  // CHECK: cl::sycl::double3 *f;
  double3 *f;
  // CHECK: double g = static_cast<double>(c.x());
  double g = c.x;
  // CHECK: a.x() = static_cast<double>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<double>(b.x()) == static_cast<double>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::double3 h[16];
  double3 h[16];
  // CHECK: cl::sycl::double3 i[32];
  double3 i[32];
  // CHECK: if (static_cast<double>(h[12].x()) == static_cast<double>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::double3 *)i;
  f = (double3 *)i;
  // CHECK: a = (cl::sycl::double3)c;
  a = (double3)c;
  // CHECK: b = cl::sycl::double3(c);
  b = double3(c);
  // CHECK: cl::sycl::double3 j, k, l, m[16], *n[32];
  double3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::double3);
  int o = sizeof(double3);
  // CHECK: int p = sizeof(double);
  int p = sizeof(double);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_double3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::double3 *e = (cl::sycl::double3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_double3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_double3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_double4(cl::sycl::double4 a, cl::sycl::double4 b, cl::sycl::double4 c) try {
void func3_double4(double4 a, double4 b, double4 c) {
}
// CHECK: void func_double4(cl::sycl::double4 a) try {
void func_double4(double4 a) {
}
// CHECK: void kernel_double4(cl::sycl::double4 *a) {
__global__ void kernel_double4(double4 *a) {
}

int main_double4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::double4 a;
  double4 a;
  // CHECK: cl::sycl::double4 b = cl::sycl::double4(1, 2, 3, 4);
  double4 b = make_double4(1, 2, 3, 4);
  // CHECK: cl::sycl::double4 c = cl::sycl::double4(b);
  double4 c = double4(b);
  // CHECK: cl::sycl::double4 d(c);
  double4 d(c);
  // CHECK: func3_double4(b, cl::sycl::double4(b), (cl::sycl::double4)b);
  func3_double4(b, double4(b), (double4)b);
  // CHECK: cl::sycl::double4 *e;
  double4 *e;
  // CHECK: cl::sycl::double4 *f;
  double4 *f;
  // CHECK: double g = static_cast<double>(c.x());
  double g = c.x;
  // CHECK: a.x() = static_cast<double>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<double>(b.x()) == static_cast<double>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::double4 h[16];
  double4 h[16];
  // CHECK: cl::sycl::double4 i[32];
  double4 i[32];
  // CHECK: if (static_cast<double>(h[12].x()) == static_cast<double>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::double4 *)i;
  f = (double4 *)i;
  // CHECK: a = (cl::sycl::double4)c;
  a = (double4)c;
  // CHECK: b = cl::sycl::double4(c);
  b = double4(c);
  // CHECK: cl::sycl::double4 j, k, l, m[16], *n[32];
  double4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::double4);
  int o = sizeof(double4);
  // CHECK: int p = sizeof(double);
  int p = sizeof(double);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_double4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::double4 *e = (cl::sycl::double4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_double4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_double4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_float1(float a, float b, float c) try {
void func3_float1(float1 a, float1 b, float1 c) {
}
// CHECK: void func_float1(float a) try {
void func_float1(float1 a) {
}
// CHECK: void kernel_float1(float *a) {
__global__ void kernel_float1(float1 *a) {
}

int main_float1() {
  // range default constructor does the right thing.
  // CHECK: float a;
  float1 a;
  // CHECK: float b = float(1);
  float1 b = make_float1(1);
  // CHECK: float c = float(b);
  float1 c = float1(b);
  // CHECK: float d(c);
  float1 d(c);
  // CHECK: func3_float1(b, float(b), (float)b);
  func3_float1(b, float1(b), (float1)b);
  // CHECK: float *e;
  float1 *e;
  // CHECK: float *f;
  float1 *f;
  // CHECK: float g = static_cast<float>(c);
  float g = c.x;
  // CHECK: a = static_cast<float>(d);
  a.x = d.x;
  // CHECK: if (static_cast<float>(b) == static_cast<float>(d)) {}
  if (b.x == d.x) {}
  // CHECK: float h[16];
  float1 h[16];
  // CHECK: float i[32];
  float1 i[32];
  // CHECK: if (static_cast<float>(h[12]) == static_cast<float>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (float *)i;
  f = (float1 *)i;
  // CHECK: a = (float)c;
  a = (float1)c;
  // CHECK: b = float(c);
  b = float1(c);
  // CHECK: float j, k, l, m[16], *n[32];
  float1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(float);
  int o = sizeof(float1);
  // CHECK: int p = sizeof(float);
  int p = sizeof(float);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_float1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           float *e = (float*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_float1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_float1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_float2(cl::sycl::float2 a, cl::sycl::float2 b, cl::sycl::float2 c) try {
void func3_float2(float2 a, float2 b, float2 c) {
}
// CHECK: void func_float2(cl::sycl::float2 a) try {
void func_float2(float2 a) {
}
// CHECK: void kernel_float2(cl::sycl::float2 *a) {
__global__ void kernel_float2(float2 *a) {
}

int main_float2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::float2 a;
  float2 a;
  // CHECK: cl::sycl::float2 b = cl::sycl::float2(1, 2);
  float2 b = make_float2(1, 2);
  // CHECK: cl::sycl::float2 c = cl::sycl::float2(b);
  float2 c = float2(b);
  // CHECK: cl::sycl::float2 d(c);
  float2 d(c);
  // CHECK: func3_float2(b, cl::sycl::float2(b), (cl::sycl::float2)b);
  func3_float2(b, float2(b), (float2)b);
  // CHECK: cl::sycl::float2 *e;
  float2 *e;
  // CHECK: cl::sycl::float2 *f;
  float2 *f;
  // CHECK: float g = static_cast<float>(c.x());
  float g = c.x;
  // CHECK: a.x() = static_cast<float>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<float>(b.x()) == static_cast<float>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::float2 h[16];
  float2 h[16];
  // CHECK: cl::sycl::float2 i[32];
  float2 i[32];
  // CHECK: if (static_cast<float>(h[12].x()) == static_cast<float>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::float2 *)i;
  f = (float2 *)i;
  // CHECK: a = (cl::sycl::float2)c;
  a = (float2)c;
  // CHECK: b = cl::sycl::float2(c);
  b = float2(c);
  // CHECK: cl::sycl::float2 j, k, l, m[16], *n[32];
  float2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::float2);
  int o = sizeof(float2);
  // CHECK: int p = sizeof(float);
  int p = sizeof(float);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_float2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::float2 *e = (cl::sycl::float2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_float2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_float2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_float3(cl::sycl::float3 a, cl::sycl::float3 b, cl::sycl::float3 c) try {
void func3_float3(float3 a, float3 b, float3 c) {
}
// CHECK: void func_float3(cl::sycl::float3 a) try {
void func_float3(float3 a) {
}
// CHECK: void kernel_float3(cl::sycl::float3 *a) {
__global__ void kernel_float3(float3 *a) {
}

int main_float3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::float3 a;
  float3 a;
  // CHECK: cl::sycl::float3 b = cl::sycl::float3(1, 2, 3);
  float3 b = make_float3(1, 2, 3);
  // CHECK: cl::sycl::float3 c = cl::sycl::float3(b);
  float3 c = float3(b);
  // CHECK: cl::sycl::float3 d(c);
  float3 d(c);
  // CHECK: func3_float3(b, cl::sycl::float3(b), (cl::sycl::float3)b);
  func3_float3(b, float3(b), (float3)b);
  // CHECK: cl::sycl::float3 *e;
  float3 *e;
  // CHECK: cl::sycl::float3 *f;
  float3 *f;
  // CHECK: float g = static_cast<float>(c.x());
  float g = c.x;
  // CHECK: a.x() = static_cast<float>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<float>(b.x()) == static_cast<float>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::float3 h[16];
  float3 h[16];
  // CHECK: cl::sycl::float3 i[32];
  float3 i[32];
  // CHECK: if (static_cast<float>(h[12].x()) == static_cast<float>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::float3 *)i;
  f = (float3 *)i;
  // CHECK: a = (cl::sycl::float3)c;
  a = (float3)c;
  // CHECK: b = cl::sycl::float3(c);
  b = float3(c);
  // CHECK: cl::sycl::float3 j, k, l, m[16], *n[32];
  float3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::float3);
  int o = sizeof(float3);
  // CHECK: int p = sizeof(float);
  int p = sizeof(float);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_float3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::float3 *e = (cl::sycl::float3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_float3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_float3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_float4(cl::sycl::float4 a, cl::sycl::float4 b, cl::sycl::float4 c) try {
void func3_float4(float4 a, float4 b, float4 c) {
}
// CHECK: void func_float4(cl::sycl::float4 a) try {
void func_float4(float4 a) {
}
// CHECK: void kernel_float4(cl::sycl::float4 *a) {
__global__ void kernel_float4(float4 *a) {
}

int main_float4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::float4 a;
  float4 a;
  // CHECK: cl::sycl::float4 b = cl::sycl::float4(1, 2, 3, 4);
  float4 b = make_float4(1, 2, 3, 4);
  // CHECK: cl::sycl::float4 c = cl::sycl::float4(b);
  float4 c = float4(b);
  // CHECK: cl::sycl::float4 d(c);
  float4 d(c);
  // CHECK: func3_float4(b, cl::sycl::float4(b), (cl::sycl::float4)b);
  func3_float4(b, float4(b), (float4)b);
  // CHECK: cl::sycl::float4 *e;
  float4 *e;
  // CHECK: cl::sycl::float4 *f;
  float4 *f;
  // CHECK: float g = static_cast<float>(c.x());
  float g = c.x;
  // CHECK: a.x() = static_cast<float>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<float>(b.x()) == static_cast<float>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::float4 h[16];
  float4 h[16];
  // CHECK: cl::sycl::float4 i[32];
  float4 i[32];
  // CHECK: if (static_cast<float>(h[12].x()) == static_cast<float>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::float4 *)i;
  f = (float4 *)i;
  // CHECK: a = (cl::sycl::float4)c;
  a = (float4)c;
  // CHECK: b = cl::sycl::float4(c);
  b = float4(c);
  // CHECK: cl::sycl::float4 j, k, l, m[16], *n[32];
  float4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::float4);
  int o = sizeof(float4);
  // CHECK: int p = sizeof(float);
  int p = sizeof(float);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_float4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::float4 *e = (cl::sycl::float4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_float4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_float4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_int1(int a, int b, int c) try {
void func3_int1(int1 a, int1 b, int1 c) {
}
// CHECK: void func_int1(int a) try {
void func_int1(int1 a) {
}
// CHECK: void kernel_int1(int *a) {
__global__ void kernel_int1(int1 *a) {
}

int main_int1() {
  // range default constructor does the right thing.
  // CHECK: int a;
  int1 a;
  // CHECK: int b = int(1);
  int1 b = make_int1(1);
  // CHECK: int c = int(b);
  int1 c = int1(b);
  // CHECK: int d(c);
  int1 d(c);
  // CHECK: func3_int1(b, int(b), (int)b);
  func3_int1(b, int1(b), (int1)b);
  // CHECK: int *e;
  int1 *e;
  // CHECK: int *f;
  int1 *f;
  // CHECK: int g = static_cast<int>(c);
  int g = c.x;
  // CHECK: a = static_cast<int>(d);
  a.x = d.x;
  // CHECK: if (static_cast<int>(b) == static_cast<int>(d)) {}
  if (b.x == d.x) {}
  // CHECK: int h[16];
  int1 h[16];
  // CHECK: int i[32];
  int1 i[32];
  // CHECK: if (static_cast<int>(h[12]) == static_cast<int>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (int *)i;
  f = (int1 *)i;
  // CHECK: a = (int)c;
  a = (int1)c;
  // CHECK: b = int(c);
  b = int1(c);
  // CHECK: int j, k, l, m[16], *n[32];
  int1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(int);
  int o = sizeof(int1);
  // CHECK: int p = sizeof(int);
  int p = sizeof(int);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_int1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           int *e = (int*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_int1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_int1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_int2(cl::sycl::int2 a, cl::sycl::int2 b, cl::sycl::int2 c) try {
void func3_int2(int2 a, int2 b, int2 c) {
}
// CHECK: void func_int2(cl::sycl::int2 a) try {
void func_int2(int2 a) {
}
// CHECK: void kernel_int2(cl::sycl::int2 *a) {
__global__ void kernel_int2(int2 *a) {
}

int main_int2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::int2 a;
  int2 a;
  // CHECK: cl::sycl::int2 b = cl::sycl::int2(1, 2);
  int2 b = make_int2(1, 2);
  // CHECK: cl::sycl::int2 c = cl::sycl::int2(b);
  int2 c = int2(b);
  // CHECK: cl::sycl::int2 d(c);
  int2 d(c);
  // CHECK: func3_int2(b, cl::sycl::int2(b), (cl::sycl::int2)b);
  func3_int2(b, int2(b), (int2)b);
  // CHECK: cl::sycl::int2 *e;
  int2 *e;
  // CHECK: cl::sycl::int2 *f;
  int2 *f;
  // CHECK: int g = static_cast<int>(c.x());
  int g = c.x;
  // CHECK: a.x() = static_cast<int>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<int>(b.x()) == static_cast<int>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::int2 h[16];
  int2 h[16];
  // CHECK: cl::sycl::int2 i[32];
  int2 i[32];
  // CHECK: if (static_cast<int>(h[12].x()) == static_cast<int>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::int2 *)i;
  f = (int2 *)i;
  // CHECK: a = (cl::sycl::int2)c;
  a = (int2)c;
  // CHECK: b = cl::sycl::int2(c);
  b = int2(c);
  // CHECK: cl::sycl::int2 j, k, l, m[16], *n[32];
  int2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::int2);
  int o = sizeof(int2);
  // CHECK: int p = sizeof(int);
  int p = sizeof(int);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_int2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::int2 *e = (cl::sycl::int2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_int2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_int2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_int3(cl::sycl::int3 a, cl::sycl::int3 b, cl::sycl::int3 c) try {
void func3_int3(int3 a, int3 b, int3 c) {
}
// CHECK: void func_int3(cl::sycl::int3 a) try {
void func_int3(int3 a) {
}
// CHECK: void kernel_int3(cl::sycl::int3 *a) {
__global__ void kernel_int3(int3 *a) {
}

int main_int3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::int3 a;
  int3 a;
  // CHECK: cl::sycl::int3 b = cl::sycl::int3(1, 2, 3);
  int3 b = make_int3(1, 2, 3);
  // CHECK: cl::sycl::int3 c = cl::sycl::int3(b);
  int3 c = int3(b);
  // CHECK: cl::sycl::int3 d(c);
  int3 d(c);
  // CHECK: func3_int3(b, cl::sycl::int3(b), (cl::sycl::int3)b);
  func3_int3(b, int3(b), (int3)b);
  // CHECK: cl::sycl::int3 *e;
  int3 *e;
  // CHECK: cl::sycl::int3 *f;
  int3 *f;
  // CHECK: int g = static_cast<int>(c.x());
  int g = c.x;
  // CHECK: a.x() = static_cast<int>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<int>(b.x()) == static_cast<int>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::int3 h[16];
  int3 h[16];
  // CHECK: cl::sycl::int3 i[32];
  int3 i[32];
  // CHECK: if (static_cast<int>(h[12].x()) == static_cast<int>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::int3 *)i;
  f = (int3 *)i;
  // CHECK: a = (cl::sycl::int3)c;
  a = (int3)c;
  // CHECK: b = cl::sycl::int3(c);
  b = int3(c);
  // CHECK: cl::sycl::int3 j, k, l, m[16], *n[32];
  int3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::int3);
  int o = sizeof(int3);
  // CHECK: int p = sizeof(int);
  int p = sizeof(int);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_int3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::int3 *e = (cl::sycl::int3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_int3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_int3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_int4(cl::sycl::int4 a, cl::sycl::int4 b, cl::sycl::int4 c) try {
void func3_int4(int4 a, int4 b, int4 c) {
}
// CHECK: void func_int4(cl::sycl::int4 a) try {
void func_int4(int4 a) {
}
// CHECK: void kernel_int4(cl::sycl::int4 *a) {
__global__ void kernel_int4(int4 *a) {
}

int main_int4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::int4 a;
  int4 a;
  // CHECK: cl::sycl::int4 b = cl::sycl::int4(1, 2, 3, 4);
  int4 b = make_int4(1, 2, 3, 4);
  // CHECK: cl::sycl::int4 c = cl::sycl::int4(b);
  int4 c = int4(b);
  // CHECK: cl::sycl::int4 d(c);
  int4 d(c);
  // CHECK: func3_int4(b, cl::sycl::int4(b), (cl::sycl::int4)b);
  func3_int4(b, int4(b), (int4)b);
  // CHECK: cl::sycl::int4 *e;
  int4 *e;
  // CHECK: cl::sycl::int4 *f;
  int4 *f;
  // CHECK: int g = static_cast<int>(c.x());
  int g = c.x;
  // CHECK: a.x() = static_cast<int>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<int>(b.x()) == static_cast<int>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::int4 h[16];
  int4 h[16];
  // CHECK: cl::sycl::int4 i[32];
  int4 i[32];
  // CHECK: if (static_cast<int>(h[12].x()) == static_cast<int>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::int4 *)i;
  f = (int4 *)i;
  // CHECK: a = (cl::sycl::int4)c;
  a = (int4)c;
  // CHECK: b = cl::sycl::int4(c);
  b = int4(c);
  // CHECK: cl::sycl::int4 j, k, l, m[16], *n[32];
  int4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::int4);
  int o = sizeof(int4);
  // CHECK: int p = sizeof(int);
  int p = sizeof(int);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_int4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::int4 *e = (cl::sycl::int4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_int4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_int4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_long1(long a, long b, long c) try {
void func3_long1(long1 a, long1 b, long1 c) {
}
// CHECK: void func_long1(long a) try {
void func_long1(long1 a) {
}
// CHECK: void kernel_long1(long *a) {
__global__ void kernel_long1(long1 *a) {
}

int main_long1() {
  // range default constructor does the right thing.
  // CHECK: long a;
  long1 a;
  // CHECK: long b = long(1);
  long1 b = make_long1(1);
  // CHECK: long c = long(b);
  long1 c = long1(b);
  // CHECK: long d(c);
  long1 d(c);
  // CHECK: func3_long1(b, long(b), (long)b);
  func3_long1(b, long1(b), (long1)b);
  // CHECK: long *e;
  long1 *e;
  // CHECK: long *f;
  long1 *f;
  // CHECK: long g = static_cast<long>(c);
  long g = c.x;
  // CHECK: a = static_cast<long>(d);
  a.x = d.x;
  // CHECK: if (static_cast<long>(b) == static_cast<long>(d)) {}
  if (b.x == d.x) {}
  // CHECK: long h[16];
  long1 h[16];
  // CHECK: long i[32];
  long1 i[32];
  // CHECK: if (static_cast<long>(h[12]) == static_cast<long>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (long *)i;
  f = (long1 *)i;
  // CHECK: a = (long)c;
  a = (long1)c;
  // CHECK: b = long(c);
  b = long1(c);
  // CHECK: long j, k, l, m[16], *n[32];
  long1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(long);
  int o = sizeof(long1);
  // CHECK: int p = sizeof(long);
  int p = sizeof(long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_long1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           long *e = (long*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_long1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_long1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_long2(cl::sycl::long2 a, cl::sycl::long2 b, cl::sycl::long2 c) try {
void func3_long2(long2 a, long2 b, long2 c) {
}
// CHECK: void func_long2(cl::sycl::long2 a) try {
void func_long2(long2 a) {
}
// CHECK: void kernel_long2(cl::sycl::long2 *a) {
__global__ void kernel_long2(long2 *a) {
}

int main_long2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::long2 a;
  long2 a;
  // CHECK: cl::sycl::long2 b = cl::sycl::long2(1, 2);
  long2 b = make_long2(1, 2);
  // CHECK: cl::sycl::long2 c = cl::sycl::long2(b);
  long2 c = long2(b);
  // CHECK: cl::sycl::long2 d(c);
  long2 d(c);
  // CHECK: func3_long2(b, cl::sycl::long2(b), (cl::sycl::long2)b);
  func3_long2(b, long2(b), (long2)b);
  // CHECK: cl::sycl::long2 *e;
  long2 *e;
  // CHECK: cl::sycl::long2 *f;
  long2 *f;
  // CHECK: long g = static_cast<long>(c.x());
  long g = c.x;
  // CHECK: a.x() = static_cast<long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<long>(b.x()) == static_cast<long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::long2 h[16];
  long2 h[16];
  // CHECK: cl::sycl::long2 i[32];
  long2 i[32];
  // CHECK: if (static_cast<long>(h[12].x()) == static_cast<long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::long2 *)i;
  f = (long2 *)i;
  // CHECK: a = (cl::sycl::long2)c;
  a = (long2)c;
  // CHECK: b = cl::sycl::long2(c);
  b = long2(c);
  // CHECK: cl::sycl::long2 j, k, l, m[16], *n[32];
  long2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::long2);
  int o = sizeof(long2);
  // CHECK: int p = sizeof(long);
  int p = sizeof(long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_long2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::long2 *e = (cl::sycl::long2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_long2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_long2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_long3(cl::sycl::long3 a, cl::sycl::long3 b, cl::sycl::long3 c) try {
void func3_long3(long3 a, long3 b, long3 c) {
}
// CHECK: void func_long3(cl::sycl::long3 a) try {
void func_long3(long3 a) {
}
// CHECK: void kernel_long3(cl::sycl::long3 *a) {
__global__ void kernel_long3(long3 *a) {
}

int main_long3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::long3 a;
  long3 a;
  // CHECK: cl::sycl::long3 b = cl::sycl::long3(1, 2, 3);
  long3 b = make_long3(1, 2, 3);
  // CHECK: cl::sycl::long3 c = cl::sycl::long3(b);
  long3 c = long3(b);
  // CHECK: cl::sycl::long3 d(c);
  long3 d(c);
  // CHECK: func3_long3(b, cl::sycl::long3(b), (cl::sycl::long3)b);
  func3_long3(b, long3(b), (long3)b);
  // CHECK: cl::sycl::long3 *e;
  long3 *e;
  // CHECK: cl::sycl::long3 *f;
  long3 *f;
  // CHECK: long g = static_cast<long>(c.x());
  long g = c.x;
  // CHECK: a.x() = static_cast<long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<long>(b.x()) == static_cast<long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::long3 h[16];
  long3 h[16];
  // CHECK: cl::sycl::long3 i[32];
  long3 i[32];
  // CHECK: if (static_cast<long>(h[12].x()) == static_cast<long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::long3 *)i;
  f = (long3 *)i;
  // CHECK: a = (cl::sycl::long3)c;
  a = (long3)c;
  // CHECK: b = cl::sycl::long3(c);
  b = long3(c);
  // CHECK: cl::sycl::long3 j, k, l, m[16], *n[32];
  long3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::long3);
  int o = sizeof(long3);
  // CHECK: int p = sizeof(long);
  int p = sizeof(long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_long3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::long3 *e = (cl::sycl::long3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_long3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_long3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_long4(cl::sycl::long4 a, cl::sycl::long4 b, cl::sycl::long4 c) try {
void func3_long4(long4 a, long4 b, long4 c) {
}
// CHECK: void func_long4(cl::sycl::long4 a) try {
void func_long4(long4 a) {
}
// CHECK: void kernel_long4(cl::sycl::long4 *a) {
__global__ void kernel_long4(long4 *a) {
}

int main_long4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::long4 a;
  long4 a;
  // CHECK: cl::sycl::long4 b = cl::sycl::long4(1, 2, 3, 4);
  long4 b = make_long4(1, 2, 3, 4);
  // CHECK: cl::sycl::long4 c = cl::sycl::long4(b);
  long4 c = long4(b);
  // CHECK: cl::sycl::long4 d(c);
  long4 d(c);
  // CHECK: func3_long4(b, cl::sycl::long4(b), (cl::sycl::long4)b);
  func3_long4(b, long4(b), (long4)b);
  // CHECK: cl::sycl::long4 *e;
  long4 *e;
  // CHECK: cl::sycl::long4 *f;
  long4 *f;
  // CHECK: long g = static_cast<long>(c.x());
  long g = c.x;
  // CHECK: a.x() = static_cast<long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<long>(b.x()) == static_cast<long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::long4 h[16];
  long4 h[16];
  // CHECK: cl::sycl::long4 i[32];
  long4 i[32];
  // CHECK: if (static_cast<long>(h[12].x()) == static_cast<long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::long4 *)i;
  f = (long4 *)i;
  // CHECK: a = (cl::sycl::long4)c;
  a = (long4)c;
  // CHECK: b = cl::sycl::long4(c);
  b = long4(c);
  // CHECK: cl::sycl::long4 j, k, l, m[16], *n[32];
  long4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::long4);
  int o = sizeof(long4);
  // CHECK: int p = sizeof(long);
  int p = sizeof(long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_long4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::long4 *e = (cl::sycl::long4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_long4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_long4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_longlong1(long long a, long long b, long long c) try {
void func3_longlong1(longlong1 a, longlong1 b, longlong1 c) {
}
// CHECK: void func_longlong1(long long a) try {
void func_longlong1(longlong1 a) {
}
// CHECK: void kernel_longlong1(long long *a) {
__global__ void kernel_longlong1(longlong1 *a) {
}

int main_longlong1() {
  // range default constructor does the right thing.
  // CHECK: long long a;
  longlong1 a;
  // CHECK: long long b = long long(1);
  longlong1 b = make_longlong1(1);
  // CHECK: long long c = long long(b);
  longlong1 c = longlong1(b);
  // CHECK: long long d(c);
  longlong1 d(c);
  // CHECK: func3_longlong1(b, long long(b), (long long)b);
  func3_longlong1(b, longlong1(b), (longlong1)b);
  // CHECK: long long *e;
  longlong1 *e;
  // CHECK: long long *f;
  longlong1 *f;
  // CHECK: long long g = static_cast<long long>(c);
  long long g = c.x;
  // CHECK: a = static_cast<long long>(d);
  a.x = d.x;
  // CHECK: if (static_cast<long long>(b) == static_cast<long long>(d)) {}
  if (b.x == d.x) {}
  // CHECK: long long h[16];
  longlong1 h[16];
  // CHECK: long long i[32];
  longlong1 i[32];
  // CHECK: if (static_cast<long long>(h[12]) == static_cast<long long>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (long long *)i;
  f = (longlong1 *)i;
  // CHECK: a = (long long)c;
  a = (longlong1)c;
  // CHECK: b = long long(c);
  b = longlong1(c);
  // CHECK: long long j, k, l, m[16], *n[32];
  longlong1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(long long);
  int o = sizeof(longlong1);
  // CHECK: int p = sizeof(long long);
  int p = sizeof(long long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_longlong1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           long long *e = (long long*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_longlong1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_longlong1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_longlong2(cl::sycl::longlong2 a, cl::sycl::longlong2 b, cl::sycl::longlong2 c) try {
void func3_longlong2(longlong2 a, longlong2 b, longlong2 c) {
}
// CHECK: void func_longlong2(cl::sycl::longlong2 a) try {
void func_longlong2(longlong2 a) {
}
// CHECK: void kernel_longlong2(cl::sycl::longlong2 *a) {
__global__ void kernel_longlong2(longlong2 *a) {
}

int main_longlong2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::longlong2 a;
  longlong2 a;
  // CHECK: cl::sycl::longlong2 b = cl::sycl::longlong2(1, 2);
  longlong2 b = make_longlong2(1, 2);
  // CHECK: cl::sycl::longlong2 c = cl::sycl::longlong2(b);
  longlong2 c = longlong2(b);
  // CHECK: cl::sycl::longlong2 d(c);
  longlong2 d(c);
  // CHECK: func3_longlong2(b, cl::sycl::longlong2(b), (cl::sycl::longlong2)b);
  func3_longlong2(b, longlong2(b), (longlong2)b);
  // CHECK: cl::sycl::longlong2 *e;
  longlong2 *e;
  // CHECK: cl::sycl::longlong2 *f;
  longlong2 *f;
  // CHECK: long long g = static_cast<long long>(c.x());
  long long g = c.x;
  // CHECK: a.x() = static_cast<long long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<long long>(b.x()) == static_cast<long long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::longlong2 h[16];
  longlong2 h[16];
  // CHECK: cl::sycl::longlong2 i[32];
  longlong2 i[32];
  // CHECK: if (static_cast<long long>(h[12].x()) == static_cast<long long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::longlong2 *)i;
  f = (longlong2 *)i;
  // CHECK: a = (cl::sycl::longlong2)c;
  a = (longlong2)c;
  // CHECK: b = cl::sycl::longlong2(c);
  b = longlong2(c);
  // CHECK: cl::sycl::longlong2 j, k, l, m[16], *n[32];
  longlong2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::longlong2);
  int o = sizeof(longlong2);
  // CHECK: int p = sizeof(long long);
  int p = sizeof(long long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_longlong2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::longlong2 *e = (cl::sycl::longlong2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_longlong2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_longlong2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_longlong3(cl::sycl::longlong3 a, cl::sycl::longlong3 b, cl::sycl::longlong3 c) try {
void func3_longlong3(longlong3 a, longlong3 b, longlong3 c) {
}
// CHECK: void func_longlong3(cl::sycl::longlong3 a) try {
void func_longlong3(longlong3 a) {
}
// CHECK: void kernel_longlong3(cl::sycl::longlong3 *a) {
__global__ void kernel_longlong3(longlong3 *a) {
}

int main_longlong3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::longlong3 a;
  longlong3 a;
  // CHECK: cl::sycl::longlong3 b = cl::sycl::longlong3(1, 2, 3);
  longlong3 b = make_longlong3(1, 2, 3);
  // CHECK: cl::sycl::longlong3 c = cl::sycl::longlong3(b);
  longlong3 c = longlong3(b);
  // CHECK: cl::sycl::longlong3 d(c);
  longlong3 d(c);
  // CHECK: func3_longlong3(b, cl::sycl::longlong3(b), (cl::sycl::longlong3)b);
  func3_longlong3(b, longlong3(b), (longlong3)b);
  // CHECK: cl::sycl::longlong3 *e;
  longlong3 *e;
  // CHECK: cl::sycl::longlong3 *f;
  longlong3 *f;
  // CHECK: long long g = static_cast<long long>(c.x());
  long long g = c.x;
  // CHECK: a.x() = static_cast<long long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<long long>(b.x()) == static_cast<long long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::longlong3 h[16];
  longlong3 h[16];
  // CHECK: cl::sycl::longlong3 i[32];
  longlong3 i[32];
  // CHECK: if (static_cast<long long>(h[12].x()) == static_cast<long long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::longlong3 *)i;
  f = (longlong3 *)i;
  // CHECK: a = (cl::sycl::longlong3)c;
  a = (longlong3)c;
  // CHECK: b = cl::sycl::longlong3(c);
  b = longlong3(c);
  // CHECK: cl::sycl::longlong3 j, k, l, m[16], *n[32];
  longlong3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::longlong3);
  int o = sizeof(longlong3);
  // CHECK: int p = sizeof(long long);
  int p = sizeof(long long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_longlong3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::longlong3 *e = (cl::sycl::longlong3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_longlong3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_longlong3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_longlong4(cl::sycl::longlong4 a, cl::sycl::longlong4 b, cl::sycl::longlong4 c) try {
void func3_longlong4(longlong4 a, longlong4 b, longlong4 c) {
}
// CHECK: void func_longlong4(cl::sycl::longlong4 a) try {
void func_longlong4(longlong4 a) {
}
// CHECK: void kernel_longlong4(cl::sycl::longlong4 *a) {
__global__ void kernel_longlong4(longlong4 *a) {
}

int main_longlong4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::longlong4 a;
  longlong4 a;
  // CHECK: cl::sycl::longlong4 b = cl::sycl::longlong4(1, 2, 3, 4);
  longlong4 b = make_longlong4(1, 2, 3, 4);
  // CHECK: cl::sycl::longlong4 c = cl::sycl::longlong4(b);
  longlong4 c = longlong4(b);
  // CHECK: cl::sycl::longlong4 d(c);
  longlong4 d(c);
  // CHECK: func3_longlong4(b, cl::sycl::longlong4(b), (cl::sycl::longlong4)b);
  func3_longlong4(b, longlong4(b), (longlong4)b);
  // CHECK: cl::sycl::longlong4 *e;
  longlong4 *e;
  // CHECK: cl::sycl::longlong4 *f;
  longlong4 *f;
  // CHECK: long long g = static_cast<long long>(c.x());
  long long g = c.x;
  // CHECK: a.x() = static_cast<long long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<long long>(b.x()) == static_cast<long long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::longlong4 h[16];
  longlong4 h[16];
  // CHECK: cl::sycl::longlong4 i[32];
  longlong4 i[32];
  // CHECK: if (static_cast<long long>(h[12].x()) == static_cast<long long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::longlong4 *)i;
  f = (longlong4 *)i;
  // CHECK: a = (cl::sycl::longlong4)c;
  a = (longlong4)c;
  // CHECK: b = cl::sycl::longlong4(c);
  b = longlong4(c);
  // CHECK: cl::sycl::longlong4 j, k, l, m[16], *n[32];
  longlong4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::longlong4);
  int o = sizeof(longlong4);
  // CHECK: int p = sizeof(long long);
  int p = sizeof(long long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_longlong4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::longlong4 *e = (cl::sycl::longlong4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_longlong4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_longlong4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_short1(short a, short b, short c) try {
void func3_short1(short1 a, short1 b, short1 c) {
}
// CHECK: void func_short1(short a) try {
void func_short1(short1 a) {
}
// CHECK: void kernel_short1(short *a) {
__global__ void kernel_short1(short1 *a) {
}

int main_short1() {
  // range default constructor does the right thing.
  // CHECK: short a;
  short1 a;
  // CHECK: short b = short(1);
  short1 b = make_short1(1);
  // CHECK: short c = short(b);
  short1 c = short1(b);
  // CHECK: short d(c);
  short1 d(c);
  // CHECK: func3_short1(b, short(b), (short)b);
  func3_short1(b, short1(b), (short1)b);
  // CHECK: short *e;
  short1 *e;
  // CHECK: short *f;
  short1 *f;
  // CHECK: short g = static_cast<short>(c);
  short g = c.x;
  // CHECK: a = static_cast<short>(d);
  a.x = d.x;
  // CHECK: if (static_cast<short>(b) == static_cast<short>(d)) {}
  if (b.x == d.x) {}
  // CHECK: short h[16];
  short1 h[16];
  // CHECK: short i[32];
  short1 i[32];
  // CHECK: if (static_cast<short>(h[12]) == static_cast<short>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (short *)i;
  f = (short1 *)i;
  // CHECK: a = (short)c;
  a = (short1)c;
  // CHECK: b = short(c);
  b = short1(c);
  // CHECK: short j, k, l, m[16], *n[32];
  short1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(short);
  int o = sizeof(short1);
  // CHECK: int p = sizeof(short);
  int p = sizeof(short);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_short1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           short *e = (short*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_short1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_short1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_short2(cl::sycl::short2 a, cl::sycl::short2 b, cl::sycl::short2 c) try {
void func3_short2(short2 a, short2 b, short2 c) {
}
// CHECK: void func_short2(cl::sycl::short2 a) try {
void func_short2(short2 a) {
}
// CHECK: void kernel_short2(cl::sycl::short2 *a) {
__global__ void kernel_short2(short2 *a) {
}

int main_short2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::short2 a;
  short2 a;
  // CHECK: cl::sycl::short2 b = cl::sycl::short2(1, 2);
  short2 b = make_short2(1, 2);
  // CHECK: cl::sycl::short2 c = cl::sycl::short2(b);
  short2 c = short2(b);
  // CHECK: cl::sycl::short2 d(c);
  short2 d(c);
  // CHECK: func3_short2(b, cl::sycl::short2(b), (cl::sycl::short2)b);
  func3_short2(b, short2(b), (short2)b);
  // CHECK: cl::sycl::short2 *e;
  short2 *e;
  // CHECK: cl::sycl::short2 *f;
  short2 *f;
  // CHECK: short g = static_cast<short>(c.x());
  short g = c.x;
  // CHECK: a.x() = static_cast<short>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<short>(b.x()) == static_cast<short>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::short2 h[16];
  short2 h[16];
  // CHECK: cl::sycl::short2 i[32];
  short2 i[32];
  // CHECK: if (static_cast<short>(h[12].x()) == static_cast<short>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::short2 *)i;
  f = (short2 *)i;
  // CHECK: a = (cl::sycl::short2)c;
  a = (short2)c;
  // CHECK: b = cl::sycl::short2(c);
  b = short2(c);
  // CHECK: cl::sycl::short2 j, k, l, m[16], *n[32];
  short2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::short2);
  int o = sizeof(short2);
  // CHECK: int p = sizeof(short);
  int p = sizeof(short);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_short2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::short2 *e = (cl::sycl::short2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_short2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_short2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_short3(cl::sycl::short3 a, cl::sycl::short3 b, cl::sycl::short3 c) try {
void func3_short3(short3 a, short3 b, short3 c) {
}
// CHECK: void func_short3(cl::sycl::short3 a) try {
void func_short3(short3 a) {
}
// CHECK: void kernel_short3(cl::sycl::short3 *a) {
__global__ void kernel_short3(short3 *a) {
}

int main_short3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::short3 a;
  short3 a;
  // CHECK: cl::sycl::short3 b = cl::sycl::short3(1, 2, 3);
  short3 b = make_short3(1, 2, 3);
  // CHECK: cl::sycl::short3 c = cl::sycl::short3(b);
  short3 c = short3(b);
  // CHECK: cl::sycl::short3 d(c);
  short3 d(c);
  // CHECK: func3_short3(b, cl::sycl::short3(b), (cl::sycl::short3)b);
  func3_short3(b, short3(b), (short3)b);
  // CHECK: cl::sycl::short3 *e;
  short3 *e;
  // CHECK: cl::sycl::short3 *f;
  short3 *f;
  // CHECK: short g = static_cast<short>(c.x());
  short g = c.x;
  // CHECK: a.x() = static_cast<short>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<short>(b.x()) == static_cast<short>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::short3 h[16];
  short3 h[16];
  // CHECK: cl::sycl::short3 i[32];
  short3 i[32];
  // CHECK: if (static_cast<short>(h[12].x()) == static_cast<short>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::short3 *)i;
  f = (short3 *)i;
  // CHECK: a = (cl::sycl::short3)c;
  a = (short3)c;
  // CHECK: b = cl::sycl::short3(c);
  b = short3(c);
  // CHECK: cl::sycl::short3 j, k, l, m[16], *n[32];
  short3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::short3);
  int o = sizeof(short3);
  // CHECK: int p = sizeof(short);
  int p = sizeof(short);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_short3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::short3 *e = (cl::sycl::short3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_short3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_short3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_short4(cl::sycl::short4 a, cl::sycl::short4 b, cl::sycl::short4 c) try {
void func3_short4(short4 a, short4 b, short4 c) {
}
// CHECK: void func_short4(cl::sycl::short4 a) try {
void func_short4(short4 a) {
}
// CHECK: void kernel_short4(cl::sycl::short4 *a) {
__global__ void kernel_short4(short4 *a) {
}

int main_short4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::short4 a;
  short4 a;
  // CHECK: cl::sycl::short4 b = cl::sycl::short4(1, 2, 3, 4);
  short4 b = make_short4(1, 2, 3, 4);
  // CHECK: cl::sycl::short4 c = cl::sycl::short4(b);
  short4 c = short4(b);
  // CHECK: cl::sycl::short4 d(c);
  short4 d(c);
  // CHECK: func3_short4(b, cl::sycl::short4(b), (cl::sycl::short4)b);
  func3_short4(b, short4(b), (short4)b);
  // CHECK: cl::sycl::short4 *e;
  short4 *e;
  // CHECK: cl::sycl::short4 *f;
  short4 *f;
  // CHECK: short g = static_cast<short>(c.x());
  short g = c.x;
  // CHECK: a.x() = static_cast<short>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<short>(b.x()) == static_cast<short>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::short4 h[16];
  short4 h[16];
  // CHECK: cl::sycl::short4 i[32];
  short4 i[32];
  // CHECK: if (static_cast<short>(h[12].x()) == static_cast<short>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::short4 *)i;
  f = (short4 *)i;
  // CHECK: a = (cl::sycl::short4)c;
  a = (short4)c;
  // CHECK: b = cl::sycl::short4(c);
  b = short4(c);
  // CHECK: cl::sycl::short4 j, k, l, m[16], *n[32];
  short4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::short4);
  int o = sizeof(short4);
  // CHECK: int p = sizeof(short);
  int p = sizeof(short);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_short4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::short4 *e = (cl::sycl::short4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_short4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_short4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_uchar1(unsigned char a, unsigned char b, unsigned char c) try {
void func3_uchar1(uchar1 a, uchar1 b, uchar1 c) {
}
// CHECK: void func_uchar1(unsigned char a) try {
void func_uchar1(uchar1 a) {
}
// CHECK: void kernel_uchar1(unsigned char *a) {
__global__ void kernel_uchar1(uchar1 *a) {
}

int main_uchar1() {
  // range default constructor does the right thing.
  // CHECK: unsigned char a;
  uchar1 a;
  // CHECK: unsigned char b = unsigned char(1);
  uchar1 b = make_uchar1(1);
  // CHECK: unsigned char c = unsigned char(b);
  uchar1 c = uchar1(b);
  // CHECK: unsigned char d(c);
  uchar1 d(c);
  // CHECK: func3_uchar1(b, unsigned char(b), (unsigned char)b);
  func3_uchar1(b, uchar1(b), (uchar1)b);
  // CHECK: unsigned char *e;
  uchar1 *e;
  // CHECK: unsigned char *f;
  uchar1 *f;
  // CHECK: unsigned char g = static_cast<unsigned char>(c);
  unsigned char g = c.x;
  // CHECK: a = static_cast<unsigned char>(d);
  a.x = d.x;
  // CHECK: if (static_cast<unsigned char>(b) == static_cast<unsigned char>(d)) {}
  if (b.x == d.x) {}
  // CHECK: unsigned char h[16];
  uchar1 h[16];
  // CHECK: unsigned char i[32];
  uchar1 i[32];
  // CHECK: if (static_cast<unsigned char>(h[12]) == static_cast<unsigned char>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (unsigned char *)i;
  f = (uchar1 *)i;
  // CHECK: a = (unsigned char)c;
  a = (uchar1)c;
  // CHECK: b = unsigned char(c);
  b = uchar1(c);
  // CHECK: unsigned char j, k, l, m[16], *n[32];
  uchar1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(unsigned char);
  int o = sizeof(uchar1);
  // CHECK: int p = sizeof(unsigned char);
  int p = sizeof(unsigned char);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_uchar1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           unsigned char *e = (unsigned char*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_uchar1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_uchar1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_uchar2(cl::sycl::uchar2 a, cl::sycl::uchar2 b, cl::sycl::uchar2 c) try {
void func3_uchar2(uchar2 a, uchar2 b, uchar2 c) {
}
// CHECK: void func_uchar2(cl::sycl::uchar2 a) try {
void func_uchar2(uchar2 a) {
}
// CHECK: void kernel_uchar2(cl::sycl::uchar2 *a) {
__global__ void kernel_uchar2(uchar2 *a) {
}

int main_uchar2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uchar2 a;
  uchar2 a;
  // CHECK: cl::sycl::uchar2 b = cl::sycl::uchar2(1, 2);
  uchar2 b = make_uchar2(1, 2);
  // CHECK: cl::sycl::uchar2 c = cl::sycl::uchar2(b);
  uchar2 c = uchar2(b);
  // CHECK: cl::sycl::uchar2 d(c);
  uchar2 d(c);
  // CHECK: func3_uchar2(b, cl::sycl::uchar2(b), (cl::sycl::uchar2)b);
  func3_uchar2(b, uchar2(b), (uchar2)b);
  // CHECK: cl::sycl::uchar2 *e;
  uchar2 *e;
  // CHECK: cl::sycl::uchar2 *f;
  uchar2 *f;
  // CHECK: unsigned char g = static_cast<unsigned char>(c.x());
  unsigned char g = c.x;
  // CHECK: a.x() = static_cast<unsigned char>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned char>(b.x()) == static_cast<unsigned char>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::uchar2 h[16];
  uchar2 h[16];
  // CHECK: cl::sycl::uchar2 i[32];
  uchar2 i[32];
  // CHECK: if (static_cast<unsigned char>(h[12].x()) == static_cast<unsigned char>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::uchar2 *)i;
  f = (uchar2 *)i;
  // CHECK: a = (cl::sycl::uchar2)c;
  a = (uchar2)c;
  // CHECK: b = cl::sycl::uchar2(c);
  b = uchar2(c);
  // CHECK: cl::sycl::uchar2 j, k, l, m[16], *n[32];
  uchar2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::uchar2);
  int o = sizeof(uchar2);
  // CHECK: int p = sizeof(unsigned char);
  int p = sizeof(unsigned char);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_uchar2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::uchar2 *e = (cl::sycl::uchar2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_uchar2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_uchar2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_uchar3(cl::sycl::uchar3 a, cl::sycl::uchar3 b, cl::sycl::uchar3 c) try {
void func3_uchar3(uchar3 a, uchar3 b, uchar3 c) {
}
// CHECK: void func_uchar3(cl::sycl::uchar3 a) try {
void func_uchar3(uchar3 a) {
}
// CHECK: void kernel_uchar3(cl::sycl::uchar3 *a) {
__global__ void kernel_uchar3(uchar3 *a) {
}

int main_uchar3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uchar3 a;
  uchar3 a;
  // CHECK: cl::sycl::uchar3 b = cl::sycl::uchar3(1, 2, 3);
  uchar3 b = make_uchar3(1, 2, 3);
  // CHECK: cl::sycl::uchar3 c = cl::sycl::uchar3(b);
  uchar3 c = uchar3(b);
  // CHECK: cl::sycl::uchar3 d(c);
  uchar3 d(c);
  // CHECK: func3_uchar3(b, cl::sycl::uchar3(b), (cl::sycl::uchar3)b);
  func3_uchar3(b, uchar3(b), (uchar3)b);
  // CHECK: cl::sycl::uchar3 *e;
  uchar3 *e;
  // CHECK: cl::sycl::uchar3 *f;
  uchar3 *f;
  // CHECK: unsigned char g = static_cast<unsigned char>(c.x());
  unsigned char g = c.x;
  // CHECK: a.x() = static_cast<unsigned char>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned char>(b.x()) == static_cast<unsigned char>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::uchar3 h[16];
  uchar3 h[16];
  // CHECK: cl::sycl::uchar3 i[32];
  uchar3 i[32];
  // CHECK: if (static_cast<unsigned char>(h[12].x()) == static_cast<unsigned char>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::uchar3 *)i;
  f = (uchar3 *)i;
  // CHECK: a = (cl::sycl::uchar3)c;
  a = (uchar3)c;
  // CHECK: b = cl::sycl::uchar3(c);
  b = uchar3(c);
  // CHECK: cl::sycl::uchar3 j, k, l, m[16], *n[32];
  uchar3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::uchar3);
  int o = sizeof(uchar3);
  // CHECK: int p = sizeof(unsigned char);
  int p = sizeof(unsigned char);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_uchar3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::uchar3 *e = (cl::sycl::uchar3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_uchar3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_uchar3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_uchar4(cl::sycl::uchar4 a, cl::sycl::uchar4 b, cl::sycl::uchar4 c) try {
void func3_uchar4(uchar4 a, uchar4 b, uchar4 c) {
}
// CHECK: void func_uchar4(cl::sycl::uchar4 a) try {
void func_uchar4(uchar4 a) {
}
// CHECK: void kernel_uchar4(cl::sycl::uchar4 *a) {
__global__ void kernel_uchar4(uchar4 *a) {
}

int main_uchar4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uchar4 a;
  uchar4 a;
  // CHECK: cl::sycl::uchar4 b = cl::sycl::uchar4(1, 2, 3, 4);
  uchar4 b = make_uchar4(1, 2, 3, 4);
  // CHECK: cl::sycl::uchar4 c = cl::sycl::uchar4(b);
  uchar4 c = uchar4(b);
  // CHECK: cl::sycl::uchar4 d(c);
  uchar4 d(c);
  // CHECK: func3_uchar4(b, cl::sycl::uchar4(b), (cl::sycl::uchar4)b);
  func3_uchar4(b, uchar4(b), (uchar4)b);
  // CHECK: cl::sycl::uchar4 *e;
  uchar4 *e;
  // CHECK: cl::sycl::uchar4 *f;
  uchar4 *f;
  // CHECK: unsigned char g = static_cast<unsigned char>(c.x());
  unsigned char g = c.x;
  // CHECK: a.x() = static_cast<unsigned char>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned char>(b.x()) == static_cast<unsigned char>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::uchar4 h[16];
  uchar4 h[16];
  // CHECK: cl::sycl::uchar4 i[32];
  uchar4 i[32];
  // CHECK: if (static_cast<unsigned char>(h[12].x()) == static_cast<unsigned char>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::uchar4 *)i;
  f = (uchar4 *)i;
  // CHECK: a = (cl::sycl::uchar4)c;
  a = (uchar4)c;
  // CHECK: b = cl::sycl::uchar4(c);
  b = uchar4(c);
  // CHECK: cl::sycl::uchar4 j, k, l, m[16], *n[32];
  uchar4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::uchar4);
  int o = sizeof(uchar4);
  // CHECK: int p = sizeof(unsigned char);
  int p = sizeof(unsigned char);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_uchar4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::uchar4 *e = (cl::sycl::uchar4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_uchar4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_uchar4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_uint1(unsigned int a, unsigned int b, unsigned int c) try {
void func3_uint1(uint1 a, uint1 b, uint1 c) {
}
// CHECK: void func_uint1(unsigned int a) try {
void func_uint1(uint1 a) {
}
// CHECK: void kernel_uint1(unsigned int *a) {
__global__ void kernel_uint1(uint1 *a) {
}

int main_uint1() {
  // range default constructor does the right thing.
  // CHECK: unsigned int a;
  uint1 a;
  // CHECK: unsigned int b = unsigned int(1);
  uint1 b = make_uint1(1);
  // CHECK: unsigned int c = unsigned int(b);
  uint1 c = uint1(b);
  // CHECK: unsigned int d(c);
  uint1 d(c);
  // CHECK: func3_uint1(b, unsigned int(b), (unsigned int)b);
  func3_uint1(b, uint1(b), (uint1)b);
  // CHECK: unsigned int *e;
  uint1 *e;
  // CHECK: unsigned int *f;
  uint1 *f;
  // CHECK: unsigned int g = static_cast<unsigned int>(c);
  unsigned int g = c.x;
  // CHECK: a = static_cast<unsigned int>(d);
  a.x = d.x;
  // CHECK: if (static_cast<unsigned int>(b) == static_cast<unsigned int>(d)) {}
  if (b.x == d.x) {}
  // CHECK: unsigned int h[16];
  uint1 h[16];
  // CHECK: unsigned int i[32];
  uint1 i[32];
  // CHECK: if (static_cast<unsigned int>(h[12]) == static_cast<unsigned int>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (unsigned int *)i;
  f = (uint1 *)i;
  // CHECK: a = (unsigned int)c;
  a = (uint1)c;
  // CHECK: b = unsigned int(c);
  b = uint1(c);
  // CHECK: unsigned int j, k, l, m[16], *n[32];
  uint1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(unsigned int);
  int o = sizeof(uint1);
  // CHECK: int p = sizeof(unsigned int);
  int p = sizeof(unsigned int);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_uint1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           unsigned int *e = (unsigned int*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_uint1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_uint1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_uint2(cl::sycl::uint2 a, cl::sycl::uint2 b, cl::sycl::uint2 c) try {
void func3_uint2(uint2 a, uint2 b, uint2 c) {
}
// CHECK: void func_uint2(cl::sycl::uint2 a) try {
void func_uint2(uint2 a) {
}
// CHECK: void kernel_uint2(cl::sycl::uint2 *a) {
__global__ void kernel_uint2(uint2 *a) {
}

int main_uint2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uint2 a;
  uint2 a;
  // CHECK: cl::sycl::uint2 b = cl::sycl::uint2(1, 2);
  uint2 b = make_uint2(1, 2);
  // CHECK: cl::sycl::uint2 c = cl::sycl::uint2(b);
  uint2 c = uint2(b);
  // CHECK: cl::sycl::uint2 d(c);
  uint2 d(c);
  // CHECK: func3_uint2(b, cl::sycl::uint2(b), (cl::sycl::uint2)b);
  func3_uint2(b, uint2(b), (uint2)b);
  // CHECK: cl::sycl::uint2 *e;
  uint2 *e;
  // CHECK: cl::sycl::uint2 *f;
  uint2 *f;
  // CHECK: unsigned int g = static_cast<unsigned int>(c.x());
  unsigned int g = c.x;
  // CHECK: a.x() = static_cast<unsigned int>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned int>(b.x()) == static_cast<unsigned int>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::uint2 h[16];
  uint2 h[16];
  // CHECK: cl::sycl::uint2 i[32];
  uint2 i[32];
  // CHECK: if (static_cast<unsigned int>(h[12].x()) == static_cast<unsigned int>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::uint2 *)i;
  f = (uint2 *)i;
  // CHECK: a = (cl::sycl::uint2)c;
  a = (uint2)c;
  // CHECK: b = cl::sycl::uint2(c);
  b = uint2(c);
  // CHECK: cl::sycl::uint2 j, k, l, m[16], *n[32];
  uint2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::uint2);
  int o = sizeof(uint2);
  // CHECK: int p = sizeof(unsigned int);
  int p = sizeof(unsigned int);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_uint2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::uint2 *e = (cl::sycl::uint2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_uint2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_uint2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_uint3(cl::sycl::uint3 a, cl::sycl::uint3 b, cl::sycl::uint3 c) try {
void func3_uint3(uint3 a, uint3 b, uint3 c) {
}
// CHECK: void func_uint3(cl::sycl::uint3 a) try {
void func_uint3(uint3 a) {
}
// CHECK: void kernel_uint3(cl::sycl::uint3 *a) {
__global__ void kernel_uint3(uint3 *a) {
}

int main_uint3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uint3 a;
  uint3 a;
  // CHECK: cl::sycl::uint3 b = cl::sycl::uint3(1, 2, 3);
  uint3 b = make_uint3(1, 2, 3);
  // CHECK: cl::sycl::uint3 c = cl::sycl::uint3(b);
  uint3 c = uint3(b);
  // CHECK: cl::sycl::uint3 d(c);
  uint3 d(c);
  // CHECK: func3_uint3(b, cl::sycl::uint3(b), (cl::sycl::uint3)b);
  func3_uint3(b, uint3(b), (uint3)b);
  // CHECK: cl::sycl::uint3 *e;
  uint3 *e;
  // CHECK: cl::sycl::uint3 *f;
  uint3 *f;
  // CHECK: unsigned int g = static_cast<unsigned int>(c.x());
  unsigned int g = c.x;
  // CHECK: a.x() = static_cast<unsigned int>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned int>(b.x()) == static_cast<unsigned int>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::uint3 h[16];
  uint3 h[16];
  // CHECK: cl::sycl::uint3 i[32];
  uint3 i[32];
  // CHECK: if (static_cast<unsigned int>(h[12].x()) == static_cast<unsigned int>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::uint3 *)i;
  f = (uint3 *)i;
  // CHECK: a = (cl::sycl::uint3)c;
  a = (uint3)c;
  // CHECK: b = cl::sycl::uint3(c);
  b = uint3(c);
  // CHECK: cl::sycl::uint3 j, k, l, m[16], *n[32];
  uint3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::uint3);
  int o = sizeof(uint3);
  // CHECK: int p = sizeof(unsigned int);
  int p = sizeof(unsigned int);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_uint3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::uint3 *e = (cl::sycl::uint3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_uint3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_uint3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_uint4(cl::sycl::uint4 a, cl::sycl::uint4 b, cl::sycl::uint4 c) try {
void func3_uint4(uint4 a, uint4 b, uint4 c) {
}
// CHECK: void func_uint4(cl::sycl::uint4 a) try {
void func_uint4(uint4 a) {
}
// CHECK: void kernel_uint4(cl::sycl::uint4 *a) {
__global__ void kernel_uint4(uint4 *a) {
}

int main_uint4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::uint4 a;
  uint4 a;
  // CHECK: cl::sycl::uint4 b = cl::sycl::uint4(1, 2, 3, 4);
  uint4 b = make_uint4(1, 2, 3, 4);
  // CHECK: cl::sycl::uint4 c = cl::sycl::uint4(b);
  uint4 c = uint4(b);
  // CHECK: cl::sycl::uint4 d(c);
  uint4 d(c);
  // CHECK: func3_uint4(b, cl::sycl::uint4(b), (cl::sycl::uint4)b);
  func3_uint4(b, uint4(b), (uint4)b);
  // CHECK: cl::sycl::uint4 *e;
  uint4 *e;
  // CHECK: cl::sycl::uint4 *f;
  uint4 *f;
  // CHECK: unsigned int g = static_cast<unsigned int>(c.x());
  unsigned int g = c.x;
  // CHECK: a.x() = static_cast<unsigned int>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned int>(b.x()) == static_cast<unsigned int>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::uint4 h[16];
  uint4 h[16];
  // CHECK: cl::sycl::uint4 i[32];
  uint4 i[32];
  // CHECK: if (static_cast<unsigned int>(h[12].x()) == static_cast<unsigned int>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::uint4 *)i;
  f = (uint4 *)i;
  // CHECK: a = (cl::sycl::uint4)c;
  a = (uint4)c;
  // CHECK: b = cl::sycl::uint4(c);
  b = uint4(c);
  // CHECK: cl::sycl::uint4 j, k, l, m[16], *n[32];
  uint4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::uint4);
  int o = sizeof(uint4);
  // CHECK: int p = sizeof(unsigned int);
  int p = sizeof(unsigned int);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_uint4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::uint4 *e = (cl::sycl::uint4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_uint4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_uint4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ulong1(unsigned long a, unsigned long b, unsigned long c) try {
void func3_ulong1(ulong1 a, ulong1 b, ulong1 c) {
}
// CHECK: void func_ulong1(unsigned long a) try {
void func_ulong1(ulong1 a) {
}
// CHECK: void kernel_ulong1(unsigned long *a) {
__global__ void kernel_ulong1(ulong1 *a) {
}

int main_ulong1() {
  // range default constructor does the right thing.
  // CHECK: unsigned long a;
  ulong1 a;
  // CHECK: unsigned long b = unsigned long(1);
  ulong1 b = make_ulong1(1);
  // CHECK: unsigned long c = unsigned long(b);
  ulong1 c = ulong1(b);
  // CHECK: unsigned long d(c);
  ulong1 d(c);
  // CHECK: func3_ulong1(b, unsigned long(b), (unsigned long)b);
  func3_ulong1(b, ulong1(b), (ulong1)b);
  // CHECK: unsigned long *e;
  ulong1 *e;
  // CHECK: unsigned long *f;
  ulong1 *f;
  // CHECK: unsigned long g = static_cast<unsigned long>(c);
  unsigned long g = c.x;
  // CHECK: a = static_cast<unsigned long>(d);
  a.x = d.x;
  // CHECK: if (static_cast<unsigned long>(b) == static_cast<unsigned long>(d)) {}
  if (b.x == d.x) {}
  // CHECK: unsigned long h[16];
  ulong1 h[16];
  // CHECK: unsigned long i[32];
  ulong1 i[32];
  // CHECK: if (static_cast<unsigned long>(h[12]) == static_cast<unsigned long>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (unsigned long *)i;
  f = (ulong1 *)i;
  // CHECK: a = (unsigned long)c;
  a = (ulong1)c;
  // CHECK: b = unsigned long(c);
  b = ulong1(c);
  // CHECK: unsigned long j, k, l, m[16], *n[32];
  ulong1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(unsigned long);
  int o = sizeof(ulong1);
  // CHECK: int p = sizeof(unsigned long);
  int p = sizeof(unsigned long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ulong1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           unsigned long *e = (unsigned long*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ulong1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ulong1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ulong2(cl::sycl::ulong2 a, cl::sycl::ulong2 b, cl::sycl::ulong2 c) try {
void func3_ulong2(ulong2 a, ulong2 b, ulong2 c) {
}
// CHECK: void func_ulong2(cl::sycl::ulong2 a) try {
void func_ulong2(ulong2 a) {
}
// CHECK: void kernel_ulong2(cl::sycl::ulong2 *a) {
__global__ void kernel_ulong2(ulong2 *a) {
}

int main_ulong2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulong2 a;
  ulong2 a;
  // CHECK: cl::sycl::ulong2 b = cl::sycl::ulong2(1, 2);
  ulong2 b = make_ulong2(1, 2);
  // CHECK: cl::sycl::ulong2 c = cl::sycl::ulong2(b);
  ulong2 c = ulong2(b);
  // CHECK: cl::sycl::ulong2 d(c);
  ulong2 d(c);
  // CHECK: func3_ulong2(b, cl::sycl::ulong2(b), (cl::sycl::ulong2)b);
  func3_ulong2(b, ulong2(b), (ulong2)b);
  // CHECK: cl::sycl::ulong2 *e;
  ulong2 *e;
  // CHECK: cl::sycl::ulong2 *f;
  ulong2 *f;
  // CHECK: unsigned long g = static_cast<unsigned long>(c.x());
  unsigned long g = c.x;
  // CHECK: a.x() = static_cast<unsigned long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned long>(b.x()) == static_cast<unsigned long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::ulong2 h[16];
  ulong2 h[16];
  // CHECK: cl::sycl::ulong2 i[32];
  ulong2 i[32];
  // CHECK: if (static_cast<unsigned long>(h[12].x()) == static_cast<unsigned long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::ulong2 *)i;
  f = (ulong2 *)i;
  // CHECK: a = (cl::sycl::ulong2)c;
  a = (ulong2)c;
  // CHECK: b = cl::sycl::ulong2(c);
  b = ulong2(c);
  // CHECK: cl::sycl::ulong2 j, k, l, m[16], *n[32];
  ulong2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::ulong2);
  int o = sizeof(ulong2);
  // CHECK: int p = sizeof(unsigned long);
  int p = sizeof(unsigned long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ulong2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::ulong2 *e = (cl::sycl::ulong2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ulong2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ulong2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ulong3(cl::sycl::ulong3 a, cl::sycl::ulong3 b, cl::sycl::ulong3 c) try {
void func3_ulong3(ulong3 a, ulong3 b, ulong3 c) {
}
// CHECK: void func_ulong3(cl::sycl::ulong3 a) try {
void func_ulong3(ulong3 a) {
}
// CHECK: void kernel_ulong3(cl::sycl::ulong3 *a) {
__global__ void kernel_ulong3(ulong3 *a) {
}

int main_ulong3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulong3 a;
  ulong3 a;
  // CHECK: cl::sycl::ulong3 b = cl::sycl::ulong3(1, 2, 3);
  ulong3 b = make_ulong3(1, 2, 3);
  // CHECK: cl::sycl::ulong3 c = cl::sycl::ulong3(b);
  ulong3 c = ulong3(b);
  // CHECK: cl::sycl::ulong3 d(c);
  ulong3 d(c);
  // CHECK: func3_ulong3(b, cl::sycl::ulong3(b), (cl::sycl::ulong3)b);
  func3_ulong3(b, ulong3(b), (ulong3)b);
  // CHECK: cl::sycl::ulong3 *e;
  ulong3 *e;
  // CHECK: cl::sycl::ulong3 *f;
  ulong3 *f;
  // CHECK: unsigned long g = static_cast<unsigned long>(c.x());
  unsigned long g = c.x;
  // CHECK: a.x() = static_cast<unsigned long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned long>(b.x()) == static_cast<unsigned long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::ulong3 h[16];
  ulong3 h[16];
  // CHECK: cl::sycl::ulong3 i[32];
  ulong3 i[32];
  // CHECK: if (static_cast<unsigned long>(h[12].x()) == static_cast<unsigned long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::ulong3 *)i;
  f = (ulong3 *)i;
  // CHECK: a = (cl::sycl::ulong3)c;
  a = (ulong3)c;
  // CHECK: b = cl::sycl::ulong3(c);
  b = ulong3(c);
  // CHECK: cl::sycl::ulong3 j, k, l, m[16], *n[32];
  ulong3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::ulong3);
  int o = sizeof(ulong3);
  // CHECK: int p = sizeof(unsigned long);
  int p = sizeof(unsigned long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ulong3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::ulong3 *e = (cl::sycl::ulong3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ulong3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ulong3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ulong4(cl::sycl::ulong4 a, cl::sycl::ulong4 b, cl::sycl::ulong4 c) try {
void func3_ulong4(ulong4 a, ulong4 b, ulong4 c) {
}
// CHECK: void func_ulong4(cl::sycl::ulong4 a) try {
void func_ulong4(ulong4 a) {
}
// CHECK: void kernel_ulong4(cl::sycl::ulong4 *a) {
__global__ void kernel_ulong4(ulong4 *a) {
}

int main_ulong4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulong4 a;
  ulong4 a;
  // CHECK: cl::sycl::ulong4 b = cl::sycl::ulong4(1, 2, 3, 4);
  ulong4 b = make_ulong4(1, 2, 3, 4);
  // CHECK: cl::sycl::ulong4 c = cl::sycl::ulong4(b);
  ulong4 c = ulong4(b);
  // CHECK: cl::sycl::ulong4 d(c);
  ulong4 d(c);
  // CHECK: func3_ulong4(b, cl::sycl::ulong4(b), (cl::sycl::ulong4)b);
  func3_ulong4(b, ulong4(b), (ulong4)b);
  // CHECK: cl::sycl::ulong4 *e;
  ulong4 *e;
  // CHECK: cl::sycl::ulong4 *f;
  ulong4 *f;
  // CHECK: unsigned long g = static_cast<unsigned long>(c.x());
  unsigned long g = c.x;
  // CHECK: a.x() = static_cast<unsigned long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned long>(b.x()) == static_cast<unsigned long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::ulong4 h[16];
  ulong4 h[16];
  // CHECK: cl::sycl::ulong4 i[32];
  ulong4 i[32];
  // CHECK: if (static_cast<unsigned long>(h[12].x()) == static_cast<unsigned long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::ulong4 *)i;
  f = (ulong4 *)i;
  // CHECK: a = (cl::sycl::ulong4)c;
  a = (ulong4)c;
  // CHECK: b = cl::sycl::ulong4(c);
  b = ulong4(c);
  // CHECK: cl::sycl::ulong4 j, k, l, m[16], *n[32];
  ulong4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::ulong4);
  int o = sizeof(ulong4);
  // CHECK: int p = sizeof(unsigned long);
  int p = sizeof(unsigned long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ulong4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::ulong4 *e = (cl::sycl::ulong4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ulong4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ulong4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ulonglong1(unsigned long long a, unsigned long long b, unsigned long long c) try {
void func3_ulonglong1(ulonglong1 a, ulonglong1 b, ulonglong1 c) {
}
// CHECK: void func_ulonglong1(unsigned long long a) try {
void func_ulonglong1(ulonglong1 a) {
}
// CHECK: void kernel_ulonglong1(unsigned long long *a) {
__global__ void kernel_ulonglong1(ulonglong1 *a) {
}

int main_ulonglong1() {
  // range default constructor does the right thing.
  // CHECK: unsigned long long a;
  ulonglong1 a;
  // CHECK: unsigned long long b = unsigned long long(1);
  ulonglong1 b = make_ulonglong1(1);
  // CHECK: unsigned long long c = unsigned long long(b);
  ulonglong1 c = ulonglong1(b);
  // CHECK: unsigned long long d(c);
  ulonglong1 d(c);
  // CHECK: func3_ulonglong1(b, unsigned long long(b), (unsigned long long)b);
  func3_ulonglong1(b, ulonglong1(b), (ulonglong1)b);
  // CHECK: unsigned long long *e;
  ulonglong1 *e;
  // CHECK: unsigned long long *f;
  ulonglong1 *f;
  // CHECK: unsigned long long g = static_cast<unsigned long long>(c);
  unsigned long long g = c.x;
  // CHECK: a = static_cast<unsigned long long>(d);
  a.x = d.x;
  // CHECK: if (static_cast<unsigned long long>(b) == static_cast<unsigned long long>(d)) {}
  if (b.x == d.x) {}
  // CHECK: unsigned long long h[16];
  ulonglong1 h[16];
  // CHECK: unsigned long long i[32];
  ulonglong1 i[32];
  // CHECK: if (static_cast<unsigned long long>(h[12]) == static_cast<unsigned long long>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (unsigned long long *)i;
  f = (ulonglong1 *)i;
  // CHECK: a = (unsigned long long)c;
  a = (ulonglong1)c;
  // CHECK: b = unsigned long long(c);
  b = ulonglong1(c);
  // CHECK: unsigned long long j, k, l, m[16], *n[32];
  ulonglong1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(unsigned long long);
  int o = sizeof(ulonglong1);
  // CHECK: int p = sizeof(unsigned long long);
  int p = sizeof(unsigned long long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ulonglong1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           unsigned long long *e = (unsigned long long*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ulonglong1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ulonglong1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ulonglong2(cl::sycl::ulonglong2 a, cl::sycl::ulonglong2 b, cl::sycl::ulonglong2 c) try {
void func3_ulonglong2(ulonglong2 a, ulonglong2 b, ulonglong2 c) {
}
// CHECK: void func_ulonglong2(cl::sycl::ulonglong2 a) try {
void func_ulonglong2(ulonglong2 a) {
}
// CHECK: void kernel_ulonglong2(cl::sycl::ulonglong2 *a) {
__global__ void kernel_ulonglong2(ulonglong2 *a) {
}

int main_ulonglong2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulonglong2 a;
  ulonglong2 a;
  // CHECK: cl::sycl::ulonglong2 b = cl::sycl::ulonglong2(1, 2);
  ulonglong2 b = make_ulonglong2(1, 2);
  // CHECK: cl::sycl::ulonglong2 c = cl::sycl::ulonglong2(b);
  ulonglong2 c = ulonglong2(b);
  // CHECK: cl::sycl::ulonglong2 d(c);
  ulonglong2 d(c);
  // CHECK: func3_ulonglong2(b, cl::sycl::ulonglong2(b), (cl::sycl::ulonglong2)b);
  func3_ulonglong2(b, ulonglong2(b), (ulonglong2)b);
  // CHECK: cl::sycl::ulonglong2 *e;
  ulonglong2 *e;
  // CHECK: cl::sycl::ulonglong2 *f;
  ulonglong2 *f;
  // CHECK: unsigned long long g = static_cast<unsigned long long>(c.x());
  unsigned long long g = c.x;
  // CHECK: a.x() = static_cast<unsigned long long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned long long>(b.x()) == static_cast<unsigned long long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::ulonglong2 h[16];
  ulonglong2 h[16];
  // CHECK: cl::sycl::ulonglong2 i[32];
  ulonglong2 i[32];
  // CHECK: if (static_cast<unsigned long long>(h[12].x()) == static_cast<unsigned long long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::ulonglong2 *)i;
  f = (ulonglong2 *)i;
  // CHECK: a = (cl::sycl::ulonglong2)c;
  a = (ulonglong2)c;
  // CHECK: b = cl::sycl::ulonglong2(c);
  b = ulonglong2(c);
  // CHECK: cl::sycl::ulonglong2 j, k, l, m[16], *n[32];
  ulonglong2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::ulonglong2);
  int o = sizeof(ulonglong2);
  // CHECK: int p = sizeof(unsigned long long);
  int p = sizeof(unsigned long long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ulonglong2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::ulonglong2 *e = (cl::sycl::ulonglong2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ulonglong2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ulonglong2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ulonglong3(cl::sycl::ulonglong3 a, cl::sycl::ulonglong3 b, cl::sycl::ulonglong3 c) try {
void func3_ulonglong3(ulonglong3 a, ulonglong3 b, ulonglong3 c) {
}
// CHECK: void func_ulonglong3(cl::sycl::ulonglong3 a) try {
void func_ulonglong3(ulonglong3 a) {
}
// CHECK: void kernel_ulonglong3(cl::sycl::ulonglong3 *a) {
__global__ void kernel_ulonglong3(ulonglong3 *a) {
}

int main_ulonglong3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulonglong3 a;
  ulonglong3 a;
  // CHECK: cl::sycl::ulonglong3 b = cl::sycl::ulonglong3(1, 2, 3);
  ulonglong3 b = make_ulonglong3(1, 2, 3);
  // CHECK: cl::sycl::ulonglong3 c = cl::sycl::ulonglong3(b);
  ulonglong3 c = ulonglong3(b);
  // CHECK: cl::sycl::ulonglong3 d(c);
  ulonglong3 d(c);
  // CHECK: func3_ulonglong3(b, cl::sycl::ulonglong3(b), (cl::sycl::ulonglong3)b);
  func3_ulonglong3(b, ulonglong3(b), (ulonglong3)b);
  // CHECK: cl::sycl::ulonglong3 *e;
  ulonglong3 *e;
  // CHECK: cl::sycl::ulonglong3 *f;
  ulonglong3 *f;
  // CHECK: unsigned long long g = static_cast<unsigned long long>(c.x());
  unsigned long long g = c.x;
  // CHECK: a.x() = static_cast<unsigned long long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned long long>(b.x()) == static_cast<unsigned long long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::ulonglong3 h[16];
  ulonglong3 h[16];
  // CHECK: cl::sycl::ulonglong3 i[32];
  ulonglong3 i[32];
  // CHECK: if (static_cast<unsigned long long>(h[12].x()) == static_cast<unsigned long long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::ulonglong3 *)i;
  f = (ulonglong3 *)i;
  // CHECK: a = (cl::sycl::ulonglong3)c;
  a = (ulonglong3)c;
  // CHECK: b = cl::sycl::ulonglong3(c);
  b = ulonglong3(c);
  // CHECK: cl::sycl::ulonglong3 j, k, l, m[16], *n[32];
  ulonglong3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::ulonglong3);
  int o = sizeof(ulonglong3);
  // CHECK: int p = sizeof(unsigned long long);
  int p = sizeof(unsigned long long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ulonglong3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::ulonglong3 *e = (cl::sycl::ulonglong3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ulonglong3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ulonglong3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ulonglong4(cl::sycl::ulonglong4 a, cl::sycl::ulonglong4 b, cl::sycl::ulonglong4 c) try {
void func3_ulonglong4(ulonglong4 a, ulonglong4 b, ulonglong4 c) {
}
// CHECK: void func_ulonglong4(cl::sycl::ulonglong4 a) try {
void func_ulonglong4(ulonglong4 a) {
}
// CHECK: void kernel_ulonglong4(cl::sycl::ulonglong4 *a) {
__global__ void kernel_ulonglong4(ulonglong4 *a) {
}

int main_ulonglong4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ulonglong4 a;
  ulonglong4 a;
  // CHECK: cl::sycl::ulonglong4 b = cl::sycl::ulonglong4(1, 2, 3, 4);
  ulonglong4 b = make_ulonglong4(1, 2, 3, 4);
  // CHECK: cl::sycl::ulonglong4 c = cl::sycl::ulonglong4(b);
  ulonglong4 c = ulonglong4(b);
  // CHECK: cl::sycl::ulonglong4 d(c);
  ulonglong4 d(c);
  // CHECK: func3_ulonglong4(b, cl::sycl::ulonglong4(b), (cl::sycl::ulonglong4)b);
  func3_ulonglong4(b, ulonglong4(b), (ulonglong4)b);
  // CHECK: cl::sycl::ulonglong4 *e;
  ulonglong4 *e;
  // CHECK: cl::sycl::ulonglong4 *f;
  ulonglong4 *f;
  // CHECK: unsigned long long g = static_cast<unsigned long long>(c.x());
  unsigned long long g = c.x;
  // CHECK: a.x() = static_cast<unsigned long long>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned long long>(b.x()) == static_cast<unsigned long long>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::ulonglong4 h[16];
  ulonglong4 h[16];
  // CHECK: cl::sycl::ulonglong4 i[32];
  ulonglong4 i[32];
  // CHECK: if (static_cast<unsigned long long>(h[12].x()) == static_cast<unsigned long long>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::ulonglong4 *)i;
  f = (ulonglong4 *)i;
  // CHECK: a = (cl::sycl::ulonglong4)c;
  a = (ulonglong4)c;
  // CHECK: b = cl::sycl::ulonglong4(c);
  b = ulonglong4(c);
  // CHECK: cl::sycl::ulonglong4 j, k, l, m[16], *n[32];
  ulonglong4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::ulonglong4);
  int o = sizeof(ulonglong4);
  // CHECK: int p = sizeof(unsigned long long);
  int p = sizeof(unsigned long long);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ulonglong4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::ulonglong4 *e = (cl::sycl::ulonglong4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ulonglong4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ulonglong4<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ushort1(unsigned short a, unsigned short b, unsigned short c) try {
void func3_ushort1(ushort1 a, ushort1 b, ushort1 c) {
}
// CHECK: void func_ushort1(unsigned short a) try {
void func_ushort1(ushort1 a) {
}
// CHECK: void kernel_ushort1(unsigned short *a) {
__global__ void kernel_ushort1(ushort1 *a) {
}

int main_ushort1() {
  // range default constructor does the right thing.
  // CHECK: unsigned short a;
  ushort1 a;
  // CHECK: unsigned short b = unsigned short(1);
  ushort1 b = make_ushort1(1);
  // CHECK: unsigned short c = unsigned short(b);
  ushort1 c = ushort1(b);
  // CHECK: unsigned short d(c);
  ushort1 d(c);
  // CHECK: func3_ushort1(b, unsigned short(b), (unsigned short)b);
  func3_ushort1(b, ushort1(b), (ushort1)b);
  // CHECK: unsigned short *e;
  ushort1 *e;
  // CHECK: unsigned short *f;
  ushort1 *f;
  // CHECK: unsigned short g = static_cast<unsigned short>(c);
  unsigned short g = c.x;
  // CHECK: a = static_cast<unsigned short>(d);
  a.x = d.x;
  // CHECK: if (static_cast<unsigned short>(b) == static_cast<unsigned short>(d)) {}
  if (b.x == d.x) {}
  // CHECK: unsigned short h[16];
  ushort1 h[16];
  // CHECK: unsigned short i[32];
  ushort1 i[32];
  // CHECK: if (static_cast<unsigned short>(h[12]) == static_cast<unsigned short>(i[12])) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (unsigned short *)i;
  f = (ushort1 *)i;
  // CHECK: a = (unsigned short)c;
  a = (ushort1)c;
  // CHECK: b = unsigned short(c);
  b = ushort1(c);
  // CHECK: unsigned short j, k, l, m[16], *n[32];
  ushort1 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(unsigned short);
  int o = sizeof(ushort1);
  // CHECK: int p = sizeof(unsigned short);
  int p = sizeof(unsigned short);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ushort1_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           unsigned short *e = (unsigned short*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ushort1(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ushort1<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ushort2(cl::sycl::ushort2 a, cl::sycl::ushort2 b, cl::sycl::ushort2 c) try {
void func3_ushort2(ushort2 a, ushort2 b, ushort2 c) {
}
// CHECK: void func_ushort2(cl::sycl::ushort2 a) try {
void func_ushort2(ushort2 a) {
}
// CHECK: void kernel_ushort2(cl::sycl::ushort2 *a) {
__global__ void kernel_ushort2(ushort2 *a) {
}

int main_ushort2() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ushort2 a;
  ushort2 a;
  // CHECK: cl::sycl::ushort2 b = cl::sycl::ushort2(1, 2);
  ushort2 b = make_ushort2(1, 2);
  // CHECK: cl::sycl::ushort2 c = cl::sycl::ushort2(b);
  ushort2 c = ushort2(b);
  // CHECK: cl::sycl::ushort2 d(c);
  ushort2 d(c);
  // CHECK: func3_ushort2(b, cl::sycl::ushort2(b), (cl::sycl::ushort2)b);
  func3_ushort2(b, ushort2(b), (ushort2)b);
  // CHECK: cl::sycl::ushort2 *e;
  ushort2 *e;
  // CHECK: cl::sycl::ushort2 *f;
  ushort2 *f;
  // CHECK: unsigned short g = static_cast<unsigned short>(c.x());
  unsigned short g = c.x;
  // CHECK: a.x() = static_cast<unsigned short>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned short>(b.x()) == static_cast<unsigned short>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::ushort2 h[16];
  ushort2 h[16];
  // CHECK: cl::sycl::ushort2 i[32];
  ushort2 i[32];
  // CHECK: if (static_cast<unsigned short>(h[12].x()) == static_cast<unsigned short>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::ushort2 *)i;
  f = (ushort2 *)i;
  // CHECK: a = (cl::sycl::ushort2)c;
  a = (ushort2)c;
  // CHECK: b = cl::sycl::ushort2(c);
  b = ushort2(c);
  // CHECK: cl::sycl::ushort2 j, k, l, m[16], *n[32];
  ushort2 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::ushort2);
  int o = sizeof(ushort2);
  // CHECK: int p = sizeof(unsigned short);
  int p = sizeof(unsigned short);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ushort2_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::ushort2 *e = (cl::sycl::ushort2*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ushort2(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ushort2<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ushort3(cl::sycl::ushort3 a, cl::sycl::ushort3 b, cl::sycl::ushort3 c) try {
void func3_ushort3(ushort3 a, ushort3 b, ushort3 c) {
}
// CHECK: void func_ushort3(cl::sycl::ushort3 a) try {
void func_ushort3(ushort3 a) {
}
// CHECK: void kernel_ushort3(cl::sycl::ushort3 *a) {
__global__ void kernel_ushort3(ushort3 *a) {
}

int main_ushort3() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ushort3 a;
  ushort3 a;
  // CHECK: cl::sycl::ushort3 b = cl::sycl::ushort3(1, 2, 3);
  ushort3 b = make_ushort3(1, 2, 3);
  // CHECK: cl::sycl::ushort3 c = cl::sycl::ushort3(b);
  ushort3 c = ushort3(b);
  // CHECK: cl::sycl::ushort3 d(c);
  ushort3 d(c);
  // CHECK: func3_ushort3(b, cl::sycl::ushort3(b), (cl::sycl::ushort3)b);
  func3_ushort3(b, ushort3(b), (ushort3)b);
  // CHECK: cl::sycl::ushort3 *e;
  ushort3 *e;
  // CHECK: cl::sycl::ushort3 *f;
  ushort3 *f;
  // CHECK: unsigned short g = static_cast<unsigned short>(c.x());
  unsigned short g = c.x;
  // CHECK: a.x() = static_cast<unsigned short>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned short>(b.x()) == static_cast<unsigned short>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::ushort3 h[16];
  ushort3 h[16];
  // CHECK: cl::sycl::ushort3 i[32];
  ushort3 i[32];
  // CHECK: if (static_cast<unsigned short>(h[12].x()) == static_cast<unsigned short>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::ushort3 *)i;
  f = (ushort3 *)i;
  // CHECK: a = (cl::sycl::ushort3)c;
  a = (ushort3)c;
  // CHECK: b = cl::sycl::ushort3(c);
  b = ushort3(c);
  // CHECK: cl::sycl::ushort3 j, k, l, m[16], *n[32];
  ushort3 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::ushort3);
  int o = sizeof(ushort3);
  // CHECK: int p = sizeof(unsigned short);
  int p = sizeof(unsigned short);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ushort3_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::ushort3 *e = (cl::sycl::ushort3*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ushort3(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ushort3<<<1,1>>>(e);
  return 0;
}

// CHECK: void func3_ushort4(cl::sycl::ushort4 a, cl::sycl::ushort4 b, cl::sycl::ushort4 c) try {
void func3_ushort4(ushort4 a, ushort4 b, ushort4 c) {
}
// CHECK: void func_ushort4(cl::sycl::ushort4 a) try {
void func_ushort4(ushort4 a) {
}
// CHECK: void kernel_ushort4(cl::sycl::ushort4 *a) {
__global__ void kernel_ushort4(ushort4 *a) {
}

int main_ushort4() {
  // range default constructor does the right thing.
  // CHECK: cl::sycl::ushort4 a;
  ushort4 a;
  // CHECK: cl::sycl::ushort4 b = cl::sycl::ushort4(1, 2, 3, 4);
  ushort4 b = make_ushort4(1, 2, 3, 4);
  // CHECK: cl::sycl::ushort4 c = cl::sycl::ushort4(b);
  ushort4 c = ushort4(b);
  // CHECK: cl::sycl::ushort4 d(c);
  ushort4 d(c);
  // CHECK: func3_ushort4(b, cl::sycl::ushort4(b), (cl::sycl::ushort4)b);
  func3_ushort4(b, ushort4(b), (ushort4)b);
  // CHECK: cl::sycl::ushort4 *e;
  ushort4 *e;
  // CHECK: cl::sycl::ushort4 *f;
  ushort4 *f;
  // CHECK: unsigned short g = static_cast<unsigned short>(c.x());
  unsigned short g = c.x;
  // CHECK: a.x() = static_cast<unsigned short>(d.x());
  a.x = d.x;
  // CHECK: if (static_cast<unsigned short>(b.x()) == static_cast<unsigned short>(d.x())) {}
  if (b.x == d.x) {}
  // CHECK: cl::sycl::ushort4 h[16];
  ushort4 h[16];
  // CHECK: cl::sycl::ushort4 i[32];
  ushort4 i[32];
  // CHECK: if (static_cast<unsigned short>(h[12].x()) == static_cast<unsigned short>(i[12].x())) {}
  if (h[12].x == i[12].x) {}
  // CHECK: f = (cl::sycl::ushort4 *)i;
  f = (ushort4 *)i;
  // CHECK: a = (cl::sycl::ushort4)c;
  a = (ushort4)c;
  // CHECK: b = cl::sycl::ushort4(c);
  b = ushort4(c);
  // CHECK: cl::sycl::ushort4 j, k, l, m[16], *n[32];
  ushort4 j, k, l, m[16], *n[32];
  // CHECK: int o = sizeof(cl::sycl::ushort4);
  int o = sizeof(ushort4);
  // CHECK: int p = sizeof(unsigned short);
  int p = sizeof(unsigned short);
  // CHECK: int q = sizeof(d);
  int q = sizeof(d);
  // CHECK: {
  // CHECK:   std::pair<syclct::buffer_t, size_t> e_buf = syclct::get_buffer_and_offset(e);
  // CHECK:   size_t e_offset = e_buf.second;
  // CHECK:   syclct::get_default_queue().submit(
  // CHECK:     [&](cl::sycl::handler &cgh) {
  // CHECK:       auto e_acc = e_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK:       cgh.parallel_for<syclct_kernel_name<class kernel_ushort4_{{[a-f0-9]+}}>>(
  // CHECK:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK:         [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK:           cl::sycl::ushort4 *e = (cl::sycl::ushort4*)(&e_acc[0] + e_offset);
  // CHECK:           kernel_ushort4(e);
  // CHECK:         });
  // CHECK:     });
  // CHECK: }
  kernel_ushort4<<<1,1>>>(e);
  return 0;
}
