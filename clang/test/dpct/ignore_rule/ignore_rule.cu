// RUN: dpct --out-root %T %s --cuda-include-path="%cuda-path/include" --rule-file %S/rule1.yaml 2>&1 | tee %T/output1.txt
// RUN: FileCheck --input-file %T/ignore_rule.dp.cpp --match-full-lines %s
// RUN: FileCheck --input-file %T/output1.txt --match-full-lines %S/output1_ref.txt
// RUN: %if build_lit %{icpx -c -fsycl %T/ignore_rule.dp.cpp -o %T/ignore_rule.dp.o %}

// CHECK: #define MACRO_A 1
// CHECK-NEXT: void foo() {
// CHECK-NEXT:   sycl::int2 a;
// CHECK-NEXT:   a.x() = MACRO_A;
// CHECK-NEXT: }
#define MACRO_A 1
void foo() {
  int2 a;
  a.x = MACRO_A;
}
