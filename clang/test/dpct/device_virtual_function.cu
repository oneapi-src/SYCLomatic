// RUN: dpct --format-range=none -out-root %T/device_virtual_function %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device_virtual_function/device_virtual_function.dp.cpp

#include <cuda_runtime.h>
template <typename T>
class TestVirtual {
public:

    __device__ TestVirtual() {}
// CHECK: /*
// CHECK-NEXT: DPCT1109:{{[0-9]+}}: Virtual functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
// CHECK-NEXT: */
    __device__ virtual ~TestVirtual() {}
// CHECK: /*
// CHECK-NEXT: DPCT1109:{{[0-9]+}}: Virtual functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
// CHECK-NEXT: */
    __device__ virtual void push(const T &&e)= 0; 
};
template <typename T>
class TestSeqContainer : public TestVirtual<T> {
public:
    __device__ TestSeqContainer(int size) : index_top(-1) { m_data = new T[size]; }

    __device__ ~TestSeqContainer() {
        if (m_data) delete []m_data;
    }
    // CHECK: /*
    // CHECK: DPCT1109:{{[0-9]+}}: Virtual functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
    // CHECK: */
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
    
    auto seq = new TestSeqContainer<int>(10);
    // CHECK: /*
    // CHECK: DPCT1109:{{[0-9]+}}: Virtual functions cannot be called in a SYCL kernel or by functions called by the kernel. You may need to adjust the code.
    // CHECK: */
    seq->push(10);
    delete seq;
}

template <class T>
class Container {
public:
    __device__ Container() {}
    __device__ ~Container() {}
};

int main() {
func<<<1,1>>>();
cudaDeviceSynchronize();
return 0;

}