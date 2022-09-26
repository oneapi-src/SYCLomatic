// RUN: dpct -in-root %S -out-root %T/convolutionbackbias %S/convolutionbackbias.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/convolutionbackbias/convolutionbackbias.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, biasTensor;
    cudnnFilterDescriptor_t filterTensor;
    auto status = cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateFilterDescriptor(&filterTensor);
    cudnnCreateTensorDescriptor(&biasTensor);
    int in = 1, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 4, ow = 4;
    int fk = 4, fc = 1, fh = 2, fw = 2;
    int ele_num = in * ic * ih * iw;
    int oele_num = on * oc * oh * ow;
    int fele_num = fk *fc * fh * fw;
    std::vector<int> bias_dim = {1, oc, 1, 1};
    std::vector<int> bias_stride = {oc, 1, 1, 1};
    int bele_num = oc * 1;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensorNdDescriptor(biasTensor, CUDNN_DATA_FLOAT, 4, bias_dim.data(), bias_stride.data());

    int filterdim[4] = {fk, fc, fh, fw};

    cudnnSetFilterNdDescriptor(filterTensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 4, filterdim);

    float *data, *out, *filter, *z, *bias;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_z(oele_num, 0.0f);
    std::vector<float> host_bias(bele_num, 0.0f);
    std::vector<float> host_filter(fk * fc * fh * fw, 0.0f);

    cudaMalloc(&data, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&z, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&bias, sizeof(float) * bele_num);
    cudaMalloc(&filter, sizeof(float) * fk * fc * fh * fw);

    cudaMemcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(z, host_z.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, host_bias.data(), sizeof(float) * bele_num, cudaMemcpyHostToDevice);

    cudaMemcpy(filter, host_filter.data(), sizeof(float) * fk * fc * fh * fw, cudaMemcpyHostToDevice);

    cudnnConvolutionDescriptor_t covdes;
    cudnnCreateConvolutionDescriptor(&covdes);
    cudnnSetConvolution2dDescriptor(covdes, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(covdes, 2);
    cudnnConvolutionFwdAlgoPerf_t algo;
    int retCount = 1;

    size_t size;
    void *workspacesize;
    // CHECK: size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, dataTensor, filterTensor, covdes, outTensor, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, &size);
    cudaMalloc(&workspacesize, size);

    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);

    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0f);
    

    float alpha = 1.f, beta = 0.f;
    // CHECK: handle.async_convolution_backward_bias(alpha, outTensor, out, beta, biasTensor, bias);
    cudnnConvolutionBackwardBias(
        handle,
        &alpha,
        outTensor,
        out,
        &beta,
        biasTensor,
        bias
    );

    cudaDeviceSynchronize();
    cudaMemcpy(host_bias.data(), bias, sizeof(float) * bele_num, cudaMemcpyDeviceToHost);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    return 0;
}