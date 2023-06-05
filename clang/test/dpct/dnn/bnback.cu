// RUN: dpct -in-root %S -out-root %T/bnback %S/bnback.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/bnback/bnback.dp.cpp --match-full-lines %s
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
    int ele_num = in* ic * ih * iw;
    int oele_num = on* oc * oh * ow;
    int sele_num = sbn*sbc * sbh * sbw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(scalebiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sbn, sbc, sbh, sbw);

    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    float *diffout, *diffdata, *diffscale, *diffbias;
    std::vector<float> host_data(ele_num, 1.0f);
    std::vector<float> host_z(ele_num, 1.0f);
    std::vector<float> host_out(oele_num, 0.0f);
    std::vector<float> host_scale(sele_num, 1.0f);
    std::vector<float> host_bias(sele_num, 0.0f);
    std::vector<float> host_rmean(sele_num, 0.0f);
    std::vector<float> host_rvar(sele_num, 0.0f);
    std::vector<float> host_smean(save * sele_num, 0.0f);
    std::vector<float> host_svar(save * sele_num, 0.0f);
    std::vector<float> host_diffout(oele_num, 0.f);
    std::vector<float> host_diffdata(ele_num, 0.f);
    std::vector<float> host_diffscale(sele_num, 1.0f);
    std::vector<float> host_diffbias(sele_num, 0.0f);


    for(int i = 0; i < ele_num; i++) {
        host_data[i] =  1.5f * i + 4.f;
    }
    for(int i = 0; i < oele_num; i++) {
        host_out[i] = 1.f;
        host_diffout[i] = 100 * i;
    }
    for(int i = 0; i < sele_num; i++) {
        host_scale[i] = i;
        host_bias[i] = i;
        host_rmean[i] = i;
        host_rvar[i] = i;
        host_smean[i] = i;
        host_svar[i] = i;
    }

    cudaMalloc(&data, sizeof(float) * ele_num);
    cudaMalloc(&z, sizeof(float) * ele_num);
    cudaMalloc(&out, sizeof(float) * oele_num);
    cudaMalloc(&scale, sizeof(float) * sele_num);
    cudaMalloc(&bias, sizeof(float) * sele_num);
    cudaMalloc(&rmean, sizeof(float) * sele_num);
    cudaMalloc(&rvar, sizeof(float) * sele_num);
    cudaMalloc(&smean, sizeof(float) * save*sele_num);
    cudaMalloc(&svar, sizeof(float)  * save*sele_num);
    cudaMalloc(&diffout, sizeof(float) * oele_num);
    cudaMalloc(&diffdata, sizeof(float) * ele_num);
    cudaMalloc(&diffscale, sizeof(float) * sele_num);
    cudaMalloc(&diffbias, sizeof(float) * sele_num);


    cudaMemcpy(data, host_data.data(), sizeof(float) * ele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(z, host_z.data(), sizeof(float) * ele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(scale, host_scale.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, host_bias.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rmean, host_rmean.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(rvar, host_rvar.data(), sizeof(float) * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(smean, host_smean.data(),  sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(svar, host_svar.data(), sizeof(float) * save * sele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), sizeof(float) * oele_num,
      cudaMemcpyHostToDevice);
    float alpha = 1.f, beta = 0.f, eps = 1.f;
    double factor = 0.1f;
// CHECK:     /*
// CHECK:     DPCT1007:{{[0-9]+}}: Migration of CUDNN_BATCHNORM_SPATIAL_PERSISTENT is not supported.
// CHECK:     */
// CHECK:     dpct::dnnl::batch_normalization_mode m = dpct::dnnl::batch_normalization_mode::spatial;
    cudnnBatchNormMode_t m = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
// CHECK: auto status = DPCT_CHECK_ERROR(handle.async_batch_normalization_forward_training(m, eps, factor, alpha, dataTensor, data, beta, outTensor, out, scalebiasTensor, scale, bias, rmean, rvar, smean, svar));
    auto status = cudnnBatchNormalizationForwardTraining(
        handle,
        m,
        &alpha,
        &beta,
        dataTensor,
        data,
        outTensor,
        out,
        scalebiasTensor,
        scale,
        bias,
        factor,
        rmean,
        rvar,
        eps,
        smean,
        svar);
// CHECK: status = DPCT_CHECK_ERROR(handle.async_batch_normalization_backward(dpct::dnnl::batch_normalization_mode::per_activation, eps, alpha, dataTensor, data, outTensor, diffout, beta, dataTensor, diffdata, alpha, scalebiasTensor, scale, beta, diffscale, diffbias, smean, svar));
    status = cudnnBatchNormalizationBackward(
        handle,
        CUDNN_BATCHNORM_PER_ACTIVATION,
        //CUDNN_BATCHNORM_SPATIAL,
        &alpha,
        &beta,
        &alpha,
        &beta,
        dataTensor,
        data,
        outTensor,
        diffout,
        dataTensor,
        diffdata,
        scalebiasTensor,
        scale,
        diffscale,
        diffbias,
        eps,
        smean,
        svar);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * oele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffdata.data(), diffdata, sizeof(float) * ele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffscale.data(), diffscale,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffbias.data(), diffbias,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    return 0;
}