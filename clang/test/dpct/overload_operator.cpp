// RUN: dpct --format-range=none --out-root %T/overload_operator %s --cuda-include-path="%cuda-path/include" --extra-arg="-xc++" || true
// RUN: FileCheck %s --match-full-lines --input-file %T/overload_operator/overload_operator.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/overload_operator/overload_operator.cpp -o %T/overload_operator/overload_operator.o %}

// CHECK: struct half {};
// CHECK-NEXT: half operator+(half &&a, half &&b) { return a; }
// CHECK-NEXT: half operator-(half &a, half &b) { return a; }
// CHECK-NEXT: void foo1() { half() + half(); }
// CHECK-NEXT: void foo2(half &a, half &b) { a - b; }
struct half {};
half operator+(half &&a, half &&b) { return a; }
half operator-(half &a, half &b) { return a; }
void foo1() { half() + half(); }
void foo2(half &a, half &b) { a - b; }
