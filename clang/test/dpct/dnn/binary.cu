// RUN: dpct -in-root %S -out-root %T/binary %S/binary.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/binary/binary.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

#define DT float
int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);

    int in = 1, ic = 1, ih = 5, iw = 5;
    int on = 1, oc = 1, oh = 5, ow = 5;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);

    DT *data, *out, *filter;
    std::vector<DT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<DT> host_out(on * oc * oh * ow, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i * 0.5f - 5.f;
    }
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }

    cudaMalloc(&data, sizeof(DT) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(DT) * on * oc * oh * ow);


    cudaMemcpy(data, host_data.data(), sizeof(DT) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(DT) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    // CHECK: dpct::dnnl::binary_op OpDesc;
    cudnnOpTensorDescriptor_t OpDesc;
    cudnnCreateOpTensorDescriptor(&OpDesc);
    // CHECK: OpDesc = dpct::dnnl::binary_op::neg;
    cudnnSetOpTensorDescriptor(OpDesc, CUDNN_OP_TENSOR_NOT, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

    float alpha0 = 1.f, alpha1 = 1.f, beta = 0.f;
    // CHECK: auto status = CHECK_SYCL_ERROR(handle.async_binary(OpDesc, alpha0, dataTensor, data, alpha1, dataTensor, data, beta, outTensor, out));
    auto status = cudnnOpTensor(
        handle, 
        OpDesc,
        &alpha0, 
        dataTensor, 
        data,
        &alpha1, 
        dataTensor, 
        data,
        &beta,
        outTensor,
        out
    );

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(DT) * on * oc * oh * ow, cudaMemcpyDeviceToHost);
    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    return 0;
}
