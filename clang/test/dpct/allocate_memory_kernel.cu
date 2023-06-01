// RUN: dpct --format-range=none -out-root %T/allocate_memory_kernel %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/allocate_memory_kernel/allocate_memory_kernel.dp.cpp
#include <cuda.h>
#include <stdio.h>

#include <cuda_runtime.h>
template <typename T>
class TestVirtualFunc {
public:

    __device__ TestVirtualFunc() {}
    __device__ virtual ~TestVirtualFunc() {}
    __device__ virtual void push(const T &&e)= 0;
};
template <typename T>
class TestSeqContainer : public TestVirtualFunc<T> {
public:
// CHECK: /*
// CHECK: DPCT1109:{{[0-9]+}}: Memory storage allocation cannot be called in SYCL device code. You need to adjust the code.
// CHECK: */
    __device__ TestSeqContainer(int size) : index_top(-1) { m_data = new T[size]; }

    __device__ ~TestSeqContainer() {
        if (m_data) delete []m_data;
    }
    __device__ virtual void push(const T &&e) {
        if (m_data) {
           int idx = atomicAdd(&this->index_top, 1);
           m_data[idx] = e;
        }
    }
private:
    T *m_data;
    int index_top;

};
__global__ void func(){
  // CHECK: /*
// CHECK: DPCT1109:{{[0-9]+}}: Memory storage allocation cannot be called in SYCL device code. You need to adjust the code.
// CHECK: */
    auto seq = new TestSeqContainer<int>(10);
    seq->push(10);
    delete seq;
}

int main() {
func<<<1,1>>>();
cudaDeviceSynchronize();
return 0;

}
