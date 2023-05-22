// RUN: dpct -in-root %S -out-root %T/reorder %S/reorder.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/reorder/reorder.dp.cpp --match-full-lines %s

// CHECK: #include <dpct/dnnl_utils.hpp>
// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include <iostream>
// CHECK: #include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

// CHECK: template <dpct::library_data_t T>
// CHECK: struct dt_trait {
// CHECK:     typedef void type;
// CHECK: };
// CHECK: template <>
// CHECK: struct dt_trait<dpct::library_data_t::real_float> {
// CHECK:     typedef float type;
// CHECK: };
// CHECK: template <>
// CHECK: /*
// CHECK: DPCT1007:{{[0-9]+}}: Migration of CUDNN_DATA_DOUBLE is not supported.
// CHECK: */
// CHECK: struct dt_trait<CUDNN_DATA_DOUBLE> {
// CHECK:     typedef double type;
// CHECK: };
// CHECK: template <>
// CHECK: struct dt_trait<dpct::library_data_t::real_int32> {
// CHECK:     typedef int type;
// CHECK: };
// CHECK: template <>
// CHECK: struct dt_trait<dpct::library_data_t::real_half> {
// CHECK:     typedef float type;
// CHECK: };

template<cudnnDataType_t T>
struct dt_trait{
    typedef void type;
};
template<>
struct dt_trait<CUDNN_DATA_FLOAT>{
    typedef float type;
};
template<>
struct dt_trait<CUDNN_DATA_DOUBLE>{
    typedef double type;
};
template<>
struct dt_trait<CUDNN_DATA_INT32>{
    typedef int type;
};
template<>
struct dt_trait<CUDNN_DATA_HALF>{
    typedef float type;
};


template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NHWC, T, n, c, h, w);

    HT *data, *out;
    std::vector<HT> host_data(ele_num, 0);
    std::vector<HT> host_out(ele_num, 0);

    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i;
        host_out[i] = 0;
    }


    cudaMalloc(&data, ele_num * sizeof(HT));
    cudaMalloc(&out, ele_num * sizeof(HT));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);


    float alpha = 3.f, beta = 1.f;
    //CHECK: auto s = CHECK_SYCL_ERROR(handle.async_reorder(alpha, dataTensor, data, beta, outTensor, out));
    auto s = cudnnTransformTensor(handle, &alpha, dataTensor, data, &beta, outTensor, out);

    cudaMemcpy(host_out.data(), out, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
}

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaSetDevice(1);
    
    test<CUDNN_DATA_FLOAT>();
       
    return 0;
}