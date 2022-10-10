// RUN: dpct -in-root %S -out-root %T/convolutionbackweight %S/convolutionbackweight.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/convolutionbackweight/convolutionbackweight.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    cudnnTensorDescriptor_t diffdataTensor, diffoutTensor;
    cudnnFilterDescriptor_t filterTensor, difffilterTensor;
    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateFilterDescriptor(&filterTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    cudnnCreateFilterDescriptor(&difffilterTensor);
    int in = 1, ic = 4, ih = 5, iw = 5;
    int on = 1, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 2, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);

    int filterdim[4] = {fk, fc, fh, fw};
    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);
    cudnnSetFilterNdDescriptor(difffilterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filterdim);

    float *data, *out, *filter, *diffdata, *diffout, *difffilter;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);
    std::vector<float> host_diffdata(in * ic * ih * iw, 1.0f);
    std::vector<float> host_diffout(on * oc * oh * ow, 0.0f);
    std::vector<float> host_difffilter(fk * fc * fh * fw, 0.0f);

    cudaMalloc(&data, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&filter, sizeof(float) * fk * fc * fh * fw);
    cudaMalloc(&diffdata, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&diffout, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&difffilter, sizeof(float) * fk * fc * fh * fw);

    cudaMemcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(difffilter, host_difffilter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(covdes, 2);

    int retCount;

    size_t size;
    void *workspacesize;
    // CHECK: size = 0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, 
        dataTensor,
        diffoutTensor, 
        covdes, 
        difffilterTensor, 
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, 
        &size);
    cudaMalloc(&workspacesize, size);

    float alpha = 1.0f, beta = 0.f;
    // CHECK: handle.async_convolution_backward_weight(covdes, dnnl::algorithm::convolution_direct, alpha, dataTensor, data, diffoutTensor, diffout, beta, difffilterTensor, difffilter);
    cudnnConvolutionBackwardFilter(
        handle, 
        &alpha,
        dataTensor,
        data,
        diffoutTensor,
        diffout, 
        covdes, 
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        workspacesize, 
        size, 
        &beta, 
        difffilterTensor, 
        difffilter);
    cudaDeviceSynchronize();
    cudaMemcpy(host_difffilter.data(), difffilter, sizeof(float) * fele_num, cudaMemcpyDeviceToHost);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    return 0;
}