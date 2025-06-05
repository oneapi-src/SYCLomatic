// RUN: dpct --format-range=none --out-root %T/register_storage %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/register_storage/register_storage.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/register_storage/register_storage.dp.cpp -o %T/register_storage/register_storage.dp.o %}

// CHECK: void foo( int b) {
// CHECK-NEXT:    int a = 5;
// CHECK-NEXT: }
void foo(register int b) {
  register int a = 5;
}
