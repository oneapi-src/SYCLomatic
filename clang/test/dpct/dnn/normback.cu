// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/normback %S/normback.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/normback/normback.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/normback/normback.dp.cpp -o %T/normback/normback.dp.o %}
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
    int sbn = 1, sbc = 4, sbh = 1, sbw = 1;
    int ele_num = in* ic * ih * iw;
    int oele_num = on* oc * oh * ow;
    int sele_num = sbn*sbc * sbh * sbw;
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(scalebiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, sbn, sbc, sbh, sbw);

    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    float *diffout, *diffdata, *diffscale, *diffbias, *diffz;
    std::vector<float> host_data(ele_num, 1.0f);
    std::vector<float> host_z(oele_num, 1.0f);
    std::vector<float> host_out(oele_num, 0.0f);
    std::vector<float> host_scale(sele_num, 1.0f);
    std::vector<float> host_bias(sele_num, 0.0f);
    std::vector<float> host_rmean(sele_num, 0.0f);
    std::vector<float> host_rvar(sele_num, 0.0f);
    std::vector<float> host_smean(save * sele_num, 0.0f);
    std::vector<float> host_svar(save * sele_num, 0.0f);
    std::vector<float> host_diffout(oele_num, 0.f);
    std::vector<float> host_diffz(oele_num, 0.f);
    std::vector<float> host_diffdata(ele_num, 0.f);
    std::vector<float> host_diffscale(sele_num, 1.0f);
    std::vector<float> host_diffbias(sele_num, 0.0f);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] =  i + 4.f;
        host_out[i] = 1.f;
        host_z[i] = 10;
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
    cudaMalloc(&z, sizeof(float) * oele_num);
    cudaMalloc(&out, sizeof(float) * oele_num);
    cudaMalloc(&scale, sizeof(float) * sele_num);
    cudaMalloc(&bias, sizeof(float) * sele_num);
    cudaMalloc(&rmean, sizeof(float) * sele_num);
    cudaMalloc(&rvar, sizeof(float) * sele_num);
    cudaMalloc(&smean, sizeof(float) * save*sele_num);
    cudaMalloc(&svar, sizeof(float)  * save*sele_num);
    cudaMalloc(&diffout, sizeof(float) * oele_num);
    cudaMalloc(&diffz, sizeof(float) * oele_num);
    cudaMalloc(&diffdata, sizeof(float) * ele_num);
    cudaMalloc(&diffscale, sizeof(float) * sele_num);
    cudaMalloc(&diffbias, sizeof(float) * sele_num);

    cudaMemcpy(data, host_data.data(), sizeof(float) * ele_num, cudaMemcpyHostToDevice);
    cudaMemcpy(z, host_z.data(), sizeof(float) * oele_num, cudaMemcpyHostToDevice);
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
    double factor = 0.5f;
    // CHECK: dpct::dnnl::activation_desc ActivationDesc;
    // CHECK: ActivationDesc.set(dnnl::algorithm::eltwise_relu_use_dst_for_bwd, 0.0f);
    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);
    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f);

    float *workspace, *reservespace;
    size_t workspace_size, reservespace_size;
    // CHECK: workspace_size = 0;
    cudnnGetNormalizationForwardTrainingWorkspaceSize(
        handle, 
        //CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        //CUDNN_NORM_OPS_NORM_ACTIVATION,
        //CUDNN_NORM_OPS_NORM_ADD_ACTIVATION,
        CUDNN_NORM_ALGO_STANDARD,
        dataTensor,
        dataTensor,
        outTensor,
        scalebiasTensor,
        ActivationDesc,
        //NULL,
        scalebiasTensor,
        &workspace_size,
        1
    );
    // CHECK: reservespace_size = handle.get_batch_normalization_workspace_size(dpct::dnnl::batch_normalization_ops::none, dataTensor);
    cudnnGetNormalizationTrainingReserveSpaceSize(
        handle,
        //CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        //CUDNN_NORM_OPS_NORM_ACTIVATION,
        //CUDNN_NORM_OPS_NORM_ADD_ACTIVATION,
        CUDNN_NORM_ALGO_STANDARD,
        NULL,
        dataTensor,
        &reservespace_size,
        1
    );
    cudaMalloc(&workspace, workspace_size);
    cudaMalloc(&reservespace,  reservespace_size);
    // CHECK: handle.async_batch_normalization_forward_training(dpct::dnnl::batch_normalization_mode::spatial, dpct::dnnl::batch_normalization_ops::none, ActivationDesc, eps, factor, alpha, dataTensor, data, beta, outTensor, out, dataTensor, z, scalebiasTensor, scale, bias, scalebiasTensor, rmean, rvar, smean, svar, reservespace_size, reservespace);
    cudnnNormalizationForwardTraining(
        handle, 
        //CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        //CUDNN_NORM_OPS_NORM_ACTIVATION,
        //CUDNN_NORM_OPS_NORM_ADD_ACTIVATION,
        CUDNN_NORM_ALGO_STANDARD,
        &alpha,
        &beta,
        dataTensor,
        data,
        scalebiasTensor,
        scale,
        bias,
        factor,
        scalebiasTensor,
        rmean,
        rvar,
        eps,
        smean,
        svar,
        ActivationDesc,
        dataTensor,
        z,
        outTensor,
        out,
        workspace,
        workspace_size,
        reservespace,
        reservespace_size,
        1);


    float *bworkspace;
    size_t bworkspace_size;
    // CHECK: bworkspace_size = 0;
    cudnnGetNormalizationForwardTrainingWorkspaceSize(
        handle, 
        //CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        //CUDNN_NORM_OPS_NORM_ACTIVATION,
        //CUDNN_NORM_OPS_NORM_ADD_ACTIVATION,
        CUDNN_NORM_ALGO_STANDARD,
        dataTensor,
        outTensor,
        outTensor,
        scalebiasTensor,
        ActivationDesc,
        scalebiasTensor,
        &bworkspace_size,
        1
    );
    cudaMalloc(&bworkspace, bworkspace_size);
    // CHECK: handle.async_batch_normalization_backward(dpct::dnnl::batch_normalization_mode::spatial, dpct::dnnl::batch_normalization_ops::none, ActivationDesc, eps, alpha, dataTensor, data, outTensor, out, outTensor, diffout, beta, dataTensor, diffdata, outTensor, diffz, alpha, scalebiasTensor, scale, bias, beta, diffscale, diffbias, scalebiasTensor, smean, svar, reservespace_size, reservespace);
    cudnnNormalizationBackward(
        handle, 
        //CUDNN_NORM_PER_ACTIVATION,
        CUDNN_NORM_PER_CHANNEL,
        CUDNN_NORM_OPS_NORM,
        //CUDNN_NORM_OPS_NORM_ACTIVATION,
        //CUDNN_NORM_OPS_NORM_ADD_ACTIVATION,
        CUDNN_NORM_ALGO_STANDARD,
        &alpha,
        &beta,
        &alpha,
        &beta,
        dataTensor,
        data,
        outTensor,
        out,
        outTensor,
        diffout,
        outTensor,
        diffz,
        dataTensor,
        diffdata,
        scalebiasTensor,
        scale,
        bias,
        diffscale,
        diffbias,
        eps,
        scalebiasTensor,
        smean,
        svar,
        ActivationDesc,
        bworkspace,
        bworkspace_size,
        reservespace,
        reservespace_size,
        1);

    cudaDeviceSynchronize();
    cudaMemcpy(host_out.data(), out, sizeof(float) * oele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_smean.data(), smean,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_svar.data(), svar,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_rmean.data(), rmean,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_rvar.data(), rvar,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffdata.data(), diffdata, sizeof(float) * ele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffz.data(), diffz, sizeof(float) * ele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffscale.data(), diffscale,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diffbias.data(), diffbias,  sizeof(float) * save * sele_num, cudaMemcpyDeviceToHost);

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    return 0;
}