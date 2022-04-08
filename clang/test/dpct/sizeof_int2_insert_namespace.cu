// RUN: c2s --format-range=none -out-root %T/sizeof_int2_insert_namespace %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/sizeof_int2_insert_namespace/sizeof_int2_insert_namespace.dp.cpp --match-full-lines %s

void fun() {
  // CHECK:  sycl::int2 a, b, c, d[2], *e[2];
  int2 a, b, c, d[2], *e[2];
  // CHECK:  int i = sizeof(sycl::int2);
  int i = sizeof(int2);
  // CHECK:  int j = sizeof(int);
  int j = sizeof(int);
  // CHECK:  sycl::int2 k;
  int2 k;
  // CHECK:  int kk = sizeof(k);
  int kk = sizeof(k);
}

