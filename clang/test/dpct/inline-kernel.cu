// RUN: dpct --format-range=none -out-root %T/nccl %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nccl/nccl.dp.cpp

// CHECK: inline void Reset_kernel_parameters(void)
__global__ void Reset_kernel_parameters(void)
{
}

// CHECK: inline void testKernel(int L, int M, const cl::sycl::nd_item<3> &[[ITEMNAME:item_ct1]], int N = 0);
__global__ void testKernel(int L, int M, int N = 0){

}

int main() {
    testKernel<<<1,2>>>(1,2,3);
}