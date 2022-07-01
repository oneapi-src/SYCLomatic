// RUN: dpct -in-root %S -out-root %T/pooling %S/pooling.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/pooling/pooling.dp.cpp --match-full-lines %s

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

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, diffdataTensor, diffoutTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);
    // CHECK: dpct::dnnl::pooling_desc desc;
    // CHECK: /*
    // CHECK: DPCT1026:{{[0-9]+}}: The call to cudnnCreatePoolingDescriptor was removed because this call is redundant in SYCL.
    // CHECK: */
    // CHECK: /*
    // CHECK: DPCT1007:{{[0-9]+}}: Migration of Nan numbers propagation option is not supported.
    // CHECK: */
    // CHECK: desc.set(dnnl::algorithm::pooling_max, 4, 4, 3, 3, 2, 2);

    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 4, 4, 3, 3, 2, 2);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;


    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);

    cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);


    int on, oc, oh, ow;
    // CHECK: desc.get_forward_output_dim(dataTensor, &on, &oc, &oh, &ow);
    cudnnGetPooling2dForwardOutputDim(desc, dataTensor, &on, &oc, &oh, &ow);

    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow);
    int ele_num2 = on * oc * oh * ow;

    HT *data, *out, *diffdata, *diffout;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num2);
    std::vector<HT> host_diffdata(ele_num);
    std::vector<HT> host_diffout(ele_num2);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i * 0.1f;
        host_diffdata[i] = i;

    }
    for(int i = 0; i < ele_num2; i++) {
        host_out[i] = i;
        host_diffout[i] = 1.f;
    }

    cudaMalloc(&data, ele_num * sizeof(HT));
    cudaMalloc(&out, ele_num2 * sizeof(HT));
    cudaMalloc(&diffdata, ele_num * sizeof(HT));
    cudaMalloc(&diffout, ele_num2 * sizeof(HT));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num2 * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), ele_num2 * sizeof(HT), cudaMemcpyHostToDevice);

    float alpha = 1.5f, beta = 1.f;
    // CHECK: handle.pooling_forward(desc, alpha, dataTensor, data, beta, outTensor, out);
    // CHECK: dpct::get_default_queue().memcpy(host_out.data(), out, ele_num2 * sizeof(HT)).wait();
    // CHECK: dpct::get_current_device().queues_wait_and_throw();
    // CHECK: /*
    // CHECK: DPCT1097:{{[0-9]+}}: The function "pooling_backward" may require the workspace which is used to save intermediate results from the "pooling_forward". By default, a workspace from engine_ext is selected according to pointer of source data, but this may be error for workspace data race. You may need to rewrite this code.
    // CHECK: */
    // CHECK: /*
    // CHECK: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK: */
    // CHECK: auto s = (handle.pooling_backward(desc, alpha, outTensor, out, diffoutTensor, diffout, dataTensor, data, beta, diffdataTensor, diffdata), 0);

    cudnnPoolingForward(handle, desc, &alpha, dataTensor, data, &beta, outTensor, out);
    cudaMemcpy(host_out.data(), out, ele_num2 * sizeof(HT), cudaMemcpyDeviceToHost);
    alpha = 1.5f, beta = 1.f;
    cudaDeviceSynchronize();
    auto s = cudnnPoolingBackward(handle, desc, &alpha, outTensor, out, diffoutTensor, diffout, dataTensor, data, &beta, diffdataTensor, diffdata);
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