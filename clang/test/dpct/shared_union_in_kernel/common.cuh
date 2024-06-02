// CHECK: void f(uint8_t *shared_storage_ct1) {
// CHECK-NEXT:   union type_ct1{ int x; };
// CHECK-NEXT:   type_ct1 &shared_storage = *(type_ct1 *)shared_storage_ct1;
// CHECK-NEXT: }
__global__ void f() {
  __shared__ union { int x; } shared_storage;
}
