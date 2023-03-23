// RUN: dpct --format-range=none -out-root %T/optimize-migration-all %s --cuda-include-path="%cuda-path/include" --usm-level=none --sycl-named-lambda --optimize-migration-all
// RUN: FileCheck %s --match-full-lines --input-file %T/optimize-migration-all/optimize-migration-all.dp.cpp

// CHECK: inline void Reset_kernel_parameters()
__global__ void Reset_kernel_parameters(void)
{
}

// CHECK: inline void testKernel(int L, int M, int N = 0){
__global__ void testKernel(int L, int M, int N = 0){

}
__global__ void my_kernel6(float* a) {}

void run_foo14(float* aa) {
//CHECK:dpct::get_default_queue().parallel_for<dpct_kernel_name<class my_kernel6_{{[0-9a-z]+}}>>(
//CHECK-NEXT:  sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:  [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:    my_kernel6((float *)nullptr);
//CHECK-NEXT:  });
  my_kernel6<<<1, 1>>>(aa);
}

int main() {
    testKernel<<<1,2>>>(1,2,3);
}