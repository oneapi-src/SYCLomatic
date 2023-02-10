#include<cuda_runtime.h>

// CHECK: void test(){
// CHECK:   int a;
// CHECK:   unsigned long long b;
// CHECK:   a++;
// CHECK: }
// CHECK: void test_host_ct{{[0-9]+}}(){
// CHECK:   int a;
// CHECK:   unsigned long long b;
// CHECK:   b++;
// CHECK: }
// CHECK: void kernel(){
// CHECK:   test();
// CHECK: }
// CHECK: int main(){
// CHECK: test_host_ct{{[0-9]+}}();
// CHECK: }
__host__ __device__ void test(){

  int a;
  unsigned long long b;
#ifdef __CUDA_ARCH__
  a++;
#else
  b++;
#endif
}

__global__ void kernel(){

  test();
}


int main(){

test();

}
