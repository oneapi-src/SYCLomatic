// RUN: dpct -in-root %S -out-root %T/activation %S/activation.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/activation/activation.dp.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include <dpct/dnnl_utils.hpp>
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
// CHECK: DPCT1007:{{[0-9]+}}: Migration of data type double is not supported.
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
    // CHECK: dpct::dnnl::engine_ext handle;
    // CHECK: dpct::dnnl::memory_desc_ext dataTensor, outTensor, diffdataTensor, diffoutTensor;

    // CHECK: handle.create_engine();

    // CHECK: sycl::queue *stream1;
    // CHECK: stream1 = dpct::get_current_device().create_queue();
    // CHECK: handle.set_queue(stream1);
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, diffdataTensor, diffoutTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);
    // CHECK: /*
    // CHECK: DPCT1026:{{[0-9]+}}: The call to cudnnCreateTensorDescriptor was removed because this call is redundant in SYCL.
    // CHECK: */
    // CHECK: /*
    // CHECK: DPCT1026:{{[0-9]+}}: The call to cudnnCreateTensorDescriptor was removed because this call is redundant in SYCL.
    // CHECK: */
    // CHECK: /*
    // CHECK: DPCT1026:{{[0-9]+}}: The call to cudnnCreateTensorDescriptor was removed because this call is redundant in SYCL.
    // CHECK: */
    // CHECK: /*
    // CHECK: DPCT1026:{{[0-9]+}}: The call to cudnnCreateTensorDescriptor was removed because this call is redundant in SYCL.
    // CHECK: */
    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    // CHECK: dataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    // CHECK: outTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    // CHECK: diffdataTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);
    // CHECK: diffoutTensor.set(dpct::dnnl::memory_format_tag::nchw, T, n, c, h, w);

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
    // CHECK: dpct::dnnl::activation_desc desc;
    // CHECK: /*
    // CHECK: DPCT1026:{{[0-9]+}}: The call to cudnnCreateActivationDescriptor was removed because this call is redundant in SYCL.
    // CHECK: */
    // CHECK: /*
    // CHECK: DPCT1007:{{[0-9]+}}: Migration of Nan numbers propagation option is not supported.
    // CHECK: */
    // CHECK: desc.set(dnnl::algorithm::eltwise_logistic_use_dst_for_bwd, 0.f);

    // CHECK: float alpha = 1.5f, beta = 0.f;
    // CHECK: handle.activation_forward(desc, alpha, dataTensor, data, beta, outTensor, out);

    // CHECK: alpha = 2.f, beta = 0.f;
    // CHECK: dpct::get_current_device().queues_wait_and_throw();
    // CHECK: /*
    // CHECK: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK: */
    // CHECK: auto s = (handle.activation_backward(desc, alpha, outTensor, out, diffoutTensor, diffout, dataTensor, data, beta, diffdataTensor, diffdata), 0);
    cudnnActivationDescriptor_t desc;
    cudnnCreateActivationDescriptor(&desc);
    cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.f);

    float alpha = 1.5f, beta = 0.f;
    cudnnActivationForward(handle, desc, &alpha, dataTensor, data, &beta, outTensor, out);

    alpha = 2.f, beta = 0.f;
    cudaDeviceSynchronize();
    auto s = cudnnActivationBackward(handle, desc, &alpha, outTensor, out, diffoutTensor, diffout, dataTensor, data, &beta, diffdataTensor, diffdata);
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
    // CHECK: test1<dpct::library_data_t::real_float>();
    test1<CUDNN_DATA_FLOAT>();

    return 0;
}