// RUN: dpct -in-root %S -out-root %T/softmax %S/softmax.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/softmax/softmax.dp.cpp --match-full-lines %s

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
void test1() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, diffdataTensor, diffoutTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    //using HT = dt_trait<T>::type;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    HT *data, *out, *diffdata, *diffout;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num);
    std::vector<HT> host_diffdata(ele_num);
    std::vector<HT> host_diffout(ele_num);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i * 0.1f;
        host_out[i] = i;
        host_diffdata[i] = i;
        host_diffout[i] = 1.f;
    }

    cudaMalloc(&data, ele_num * sizeof(HT));
    cudaMalloc(&out, ele_num * sizeof(HT));
    cudaMalloc(&diffdata, ele_num * sizeof(HT));
    cudaMalloc(&diffout, ele_num * sizeof(HT));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);

    float alpha = 1.5f, beta = 0.f;
    // CHECK: handle.async_softmax_forward(dpct::dnnl::softmax_algorithm::normal, dpct::dnnl::softmax_mode::channel, alpha, dataTensor, data, beta, outTensor, out);
    // CHECK: dpct::get_default_queue().memcpy(host_out.data(), out, ele_num * sizeof(HT)).wait();
    // CHECK: alpha = 2.f, beta = 0.f;
    // CHECK: dpct::get_current_device().queues_wait_and_throw();
    // CHECK: auto s = DPCT_CHECK_ERROR(handle.async_softmax_backward(dpct::dnnl::softmax_algorithm::normal, dpct::dnnl::softmax_mode::channel, alpha, outTensor, out, diffoutTensor, diffout, beta, diffdataTensor, diffdata));

    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, dataTensor, data, &beta, outTensor, out);
    cudaMemcpy(host_out.data(), out, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);
    alpha = 2.f, beta = 0.f;
    cudaDeviceSynchronize();
    auto s = cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, outTensor, out, diffoutTensor, diffout, &beta, diffdataTensor, diffdata);
    cudaDeviceSynchronize();

    cudaMemcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    cudaFree(diffdata);
    cudaFree(diffout);
}

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaSetDevice(1);
    
    test1<CUDNN_DATA_FLOAT>();

    return 0;
}