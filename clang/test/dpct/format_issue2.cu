// RUN: c2s -out-root %T/format_issue2 %s --cuda-include-path="%cuda-path/include" --  -x cuda --cuda-host-only -DCOMPILATION
// RUN: FileCheck %s --match-full-lines --input-file %T/format_issue2/format_issue2.dp.cpp

float a;int
#define AAA
#define BBB
#define CCC
#define DDD
#define EEE
#define FFF
foo1(){ return 1; }
int main(){ int2 i; }


// CHECK: float a;int
// CHECK-NEXT: #define AAA
// CHECK-NEXT: #define BBB
// CHECK-NEXT: #define CCC
// CHECK-NEXT: #define DDD
// CHECK-NEXT: #define EEE
// CHECK-NEXT: #define FFF
// CHECK-NEXT: foo1(){ return 1; }
// CHECK-NEXT: int main() { sycl::int2 i; }