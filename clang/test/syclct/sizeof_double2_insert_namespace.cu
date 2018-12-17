// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/sizeof_double2_insert_namespace.sycl.cpp --match-full-lines %s

void fun() {
  // CHECK:  int i = sizeof(cl::sycl::double2);
  int i = sizeof(double2);
  // CHECK:  int j = sizeof(int);
  int j = sizeof(int);
  // CHECK:  cl::sycl::double2 k;
  double2 k;
  // CHECK:  int kk = sizeof(k);
  int kk = sizeof(k);
}
