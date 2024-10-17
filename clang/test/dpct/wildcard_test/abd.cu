// RUN: dpct --format-range=none --usm-level=none -out-root=%T/abd -in-root=%S %S/ab*.cu --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/abd/abd.dp.cpp --match-full-lines %S/abd.cu
// RUN: %if build_lit %{icpx -c -fsycl %T/abd/abd.dp.cpp -o %T/abd/abd.dp.o %}
// RUN: FileCheck --input-file %T/abd/abc.dp.cpp --match-full-lines %S/abc.cu
// RUN: %if build_lit %{icpx -c -fsycl %T/abd/abc.dp.cpp -o %T/abd/abc.dp.o %}

int printf(const char *format, ...);

const char *test_function() {

//CHECK:/*
//CHECK-NEXT:DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", dpct::get_error_string_dummy(0));
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
