// RUN: dpct --format-range=none -in-root %S -out-root %T/bninfer %S/bninfer.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/bninfer/bninfer.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, scalebiasTensor;
    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&scalebiasTensor);

    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 5, ow = 5;
    int sbn = 1, sbc = 4, sbh = 5, sbw = 5;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(scalebiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sbn, sbc, sbh, sbw);

    // CHECK: dpct::dnnl::derive_batch_normalization_memory_desc(outTensor, dataTensor, dpct::dnnl::batch_normalization_mode::per_activation);
    cudnnDeriveBNTensorDescriptor(outTensor, dataTensor, CUDNN_BATCHNORM_PER_ACTIVATION);

    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    std::vector<float> host_data(in * ic * ih * iw, 1.0f);
    std::vector<float> host_z(in * ic * ih * iw, 1.0f);
    std::vector<float> host_out(on * oc * oh * ow, 0.0f);
    std::vector<float> host_scale(sbn * sbc * sbh * sbw, 1.0f);
    std::vector<float> host_bias(sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_rmean(sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_rvar(sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_smean(save * sbn * sbc * sbh * sbw, 0.0f);
    std::vector<float> host_svar(save * sbn * sbc * sbh * sbw, 0.0f);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] =  i + 4.f;
        host_out[i] = 1.f;
        host_z[i] = 10;
    }
    for(int i = 0; i < sbn * sbc * sbh * sbw; i++) {
        host_scale[i] = i;
        host_bias[i] = i;
        host_smean[i] = i;
        host_svar[i] = i;
    }

    cudaMalloc(&data, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&z, sizeof(float) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(float) * on * oc * oh * ow);
    cudaMalloc(&scale, sizeof(float) * sbn * sbc * sbh * sbw);
    cudaMalloc(&bias, sizeof(float) * sbn * sbc * sbh * sbw);
    cudaMalloc(&rmean, sizeof(float) * sbn * sbc * sbh * sbw);
    cudaMalloc(&rvar, sizeof(float) * sbn * sbc * sbh * sbw);
    cudaMalloc(&smean, sizeof(float) * save*sbn * sbc * sbh * sbw);
    cudaMalloc(&svar, sizeof(float)  * save*sbn * sbc * sbh * sbw);

    cudaMemcpy(data, host_data.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(z, host_z.data(), sizeof(float) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * on * oc * oh * ow, cudaMemcpyHostToDevice);
    cudaMemcpy(scale, host_scale.data(), sizeof(float) * sbn * sbc * sbh * sbw, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, host_bias.data(), sizeof(float) * sbn * sbc * sbh * sbw, cudaMemcpyHostToDevice);
    cudaMemcpy(rmean, host_rmean.data(), sizeof(float) * sbn * sbc * sbh * sbw, cudaMemcpyHostToDevice);
    cudaMemcpy(rvar, host_rvar.data(), sizeof(float) * sbn * sbc * sbh * sbw, cudaMemcpyHostToDevice);
    cudaMemcpy(smean, host_smean.data(),  sizeof(float) * save * sbn * sbc * sbh * sbw, cudaMemcpyHostToDevice);
    cudaMemcpy(svar, host_svar.data(), sizeof(float) * save * sbn * sbc * sbh * sbw, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.f, eps = 1.f;
    // CHECK: auto status = DPCT_CHECK_ERROR(handle.async_batch_normalization_forward_inference(dpct::dnnl::batch_normalization_mode::per_activation, eps, alpha, dataTensor, data, beta, outTensor, out, scalebiasTensor, scale, bias, smean, svar));
    auto status = cudnnBatchNormalizationForwardInference(
        handle,
        CUDNN_BATCHNORM_PER_ACTIVATION,
        &alpha,
        &beta,
        dataTensor,
        data,
        outTensor,
        out,
        scalebiasTensor,
        scale,
        bias,
        smean,
        svar,
        eps);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * on * oc * oh * ow, cudaMemcpyDeviceToHost);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    return 0;
}