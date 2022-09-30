// RUN: dpct -in-root %S -out-root %T/convolution %S/convolution.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/convolution/convolution.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    cudnnFilterDescriptor_t filterTensor;
    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateFilterDescriptor(&filterTensor);
    int in = 1, ic = 2, ih = 5, iw = 5;
    int on = 1, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 2, fh = 2, fw = 2;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    
    int filterdim[4] = {fk, fc, fh, fw};
    // CHECK: filterTensor.set(dpct::dnnl::memory_format_tag::nchw, dpct::library_data_t::real_float, 4, filterdim);
    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);
    
    float *data, *out, *filter, *bias;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_bias(on * oc * oh * ow, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);


    cudaMalloc(&data, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&bias, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&filter, sizeof(float) * fk * fc * fh * fw);

    cudaMemcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, host_bias.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);
    // CHECK: dpct::dnnl::convolution_desc covdes;
    // CHECK: covdes.set(0, 0, 1, 1, 1, 1);
    // CHECK: covdes.set_group_count(2);
    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(covdes, 2);
    // CHECK: /*
    // CHECK: DPCT1007:{{[0-9]+}}: Migration of CUDNN_CONVOLUTION is not supported.
    // CHECK: */
    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;
    int retCount;
    size_t size;
    void *workspacesize;
    // CHECK: size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, dataTensor, filterTensor, covdes, outTensor, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, &size);
    cudaMalloc(&workspacesize, size);

    int dimo[4];
    // CHECK: covdes.get_forward_output_dim(dataTensor, filterTensor, 4, dimo);
    cudnnGetConvolutionNdForwardOutputDim(covdes, dataTensor, filterTensor, 4, dimo);

    float alpha = 1.0f, beta = 0.0f;
    // CHECK: handle.async_convolution_forward(covdes, dnnl::algorithm::convolution_direct, alpha, dataTensor, data, filterTensor, filter, beta, outTensor, out);
    cudnnConvolutionForward(handle, &alpha, dataTensor, data, filterTensor, filter, covdes, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, workspacesize, size, &beta, outTensor, out);

    cudaDeviceSynchronize();
    cudaMemcpy(host_bias.data(), bias, sizeof(float) * on * oc * oh * ow, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out.data(), out, sizeof(float) * on * oc * oh * ow, cudaMemcpyDeviceToHost);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    return 0;
}