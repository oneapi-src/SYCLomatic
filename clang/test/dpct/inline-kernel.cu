// RUN: dpct --format-range=none -out-root %T/inline-kernel %s --cuda-include-path="%cuda-path/include" --opt-migration=inline-kernel-function
// RUN: FileCheck %s --match-full-lines --input-file %T/inline-kernel/inline-kernel.dp.cpp

// CHECK: inline void Reset_kernel_parameters()
__global__ void Reset_kernel_parameters(void)
{
}

// CHECK: inline void testKernel(int L, int M, int N = 0){
__global__ void testKernel(int L, int M, int N = 0){

}

int main() {
    testKernel<<<1,2>>>(1,2,3);
}