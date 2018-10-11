// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sizeof_int2_insert_namespace.sycl.cpp --match-full-lines %s

void fun() {
  // CHECK:  int i = sizeof(cl::sycl::int2);
  int i = sizeof(int2);
  // CHECK:  int j = sizeof(int);
  int j = sizeof(int);
  // CHECK:  cl::sycl::int2 k;
  int2 k;
  // CHECK:  int kk = sizeof(k);
  int kk = sizeof(k);
}
