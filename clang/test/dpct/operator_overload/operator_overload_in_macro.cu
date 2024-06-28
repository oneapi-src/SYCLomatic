// RUN: dpct -out-root %T/operator_overload/operator_overload_in_macro %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/operator_overload/operator_overload_in_macro/operator_overload_in_macro.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/operator_overload/operator_overload_in_macro/operator_overload_in_macro.dp.cpp -o %T/operator_overload/operator_overload_in_macro/operator_overload_in_macro.dp.o %}

#define OPMACRO(operation, ...) \
  bool operation(__VA_ARGS__) { return true; }

//CHECK: /*
//CHECK-NEXT: DPCT1011:{{[0-9]+}}: The tool detected overloaded operators for built-in vector types, which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020 standard operators instead.
//CHECK-NEXT: */
//CHECK-NEXT: namespace dpct_operator_overloading {
//CHECK-EMPTY:
//CHECK-NEXT: OPMACRO(operator-, sycl::double2 v)
//CHECK-NEXT: }  // namespace dpct_operator_overloading
OPMACRO(operator-, double2 v)

int main() {
  double2 v;
  //CHECK: dpct_operator_overloading::operator-(v);
  -v;
  return 0;
}
