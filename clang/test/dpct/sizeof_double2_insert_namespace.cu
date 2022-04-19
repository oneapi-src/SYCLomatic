// RUN: c2s --format-range=none -out-root %T/sizeof_double2_insert_namespace %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sizeof_double2_insert_namespace/sizeof_double2_insert_namespace.dp.cpp --match-full-lines %s

void fun() {
  // CHECK:  int i = sizeof(sycl::double2);
  int i = sizeof(double2);
  // CHECK:  int j = sizeof(int);
  int j = sizeof(int);
  // CHECK:  sycl::double2 k;
  double2 k;
  // CHECK:  int kk = sizeof(k);
  int kk = sizeof(k);
}

