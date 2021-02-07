// RUN: cd %T
// RUN: dpct --format-range=none --no-cl-namespace-inline --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -fno-delayed-template-parsing -std=c++14 2>err.txt
// RUN: FileCheck --input-file %T/unique_id.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/err.txt --match-full-lines %S/ref.txt


//CHECK:/*
//CHECK-NEXT:DPCT1054:0: The type of variable temp is declared in device function with the name type_ct1. Adjust the code to make the type_ct1 declaration visible at the accessor declaration point.
//CHECK-NEXT:*/
//CHECK:/*
//CHECK-NEXT:DPCT1054:1: The type of variable temp2 is declared in device function with the name type_ct1. Adjust the code to make the type_ct1 declaration visible at the accessor declaration point.
//CHECK-NEXT:*/
template <typename T>
__global__ void k(T a){
__shared__ union {
    T up;
  } temp, temp2;
  temp.up = a;
  temp2.up = a;
}
template<typename TT>
void foo() {
  TT a;
  k<TT><<<1,1>>>(a);
}
