// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test20_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test20_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test20_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test20_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test20_out

// CHECK: 36
// TEST_FEATURE: DnnlUtils_batch_normalization_backward_ex_norm
// TEST_FEATURE: DnnlUtils_batch_normalization_mode
// TEST_FEATURE: DnnlUtils_batch_normalization_ops
// TEST_FEATURE: DnnlUtils_get_batch_normalization_workspace_size

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    int nDevices;
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, scalebiasTensor;
    cudaStream_t stream1;
    int in = 2, ic = 4, ih = 5, iw = 5;
    int on = 2, oc = 4, oh = 5, ow = 5;
    int sbn = 1, sbc = 4, sbh = 1, sbw = 1;
    int ele_num = in* ic * ih * iw;
    int oele_num = on* oc * oh * ow;
    int sele_num = sbn*sbc * sbh * sbw;
    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    float *diffout, *diffdata, *diffscale, *diffbias, *diffz;
    float alpha = 1.f, beta = 0.f, eps = 1.f;
    double factor = 0.5f;

    cudnnActivationDescriptor_t ActivationDesc;
    cudnnCreateActivationDescriptor(&ActivationDesc);
    cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f);

    float *workspace, *reservespace;
    size_t workspace_size, reservespace_size;
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

    return 0;
}