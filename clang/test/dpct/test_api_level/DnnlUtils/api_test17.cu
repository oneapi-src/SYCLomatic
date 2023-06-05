// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test17_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test17_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test17_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test17_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test17_out

// CHECK: 21
// TEST_FEATURE: DnnlUtils_batch_normalization_backward_ex
// TEST_FEATURE: DnnlUtils_batch_normalization_mode
// TEST_FEATURE: DnnlUtils_batch_normalization_ops
// TEST_FEATURE: DnnlUtils_get_batch_normalization_workspace_size

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

    cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        handle, 
        CUDNN_BATCHNORM_PER_ACTIVATION, 
        //CUDNN_BATCHNORM_SPATIAL, 
        CUDNN_BATCHNORM_OPS_BN_ACTIVATION, 
        //CUDNN_BATCHNORM_OPS_BN,
        //CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION, 
        dataTensor,
        outTensor, 
        outTensor, 
        scalebiasTensor, 
        ActivationDesc, 
        &workspace_size);

    cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        handle, 
        CUDNN_BATCHNORM_PER_ACTIVATION, 
        //CUDNN_BATCHNORM_SPATIAL, 
        CUDNN_BATCHNORM_OPS_BN_ACTIVATION, 
        //CUDNN_BATCHNORM_OPS_BN,
        //CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION, 
        ActivationDesc, 
        dataTensor, 
        &reservespace_size);

    auto status = cudnnBatchNormalizationForwardTrainingEx(
        handle, 
        CUDNN_BATCHNORM_PER_ACTIVATION, 
        //CUDNN_BATCHNORM_SPATIAL, 
        CUDNN_BATCHNORM_OPS_BN_ACTIVATION, 
        //CUDNN_BATCHNORM_OPS_BN,
        //CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION,
        &alpha,
        &beta,
        dataTensor,
        data,
        outTensor,
        z,
        //nullptr,
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
        svar,
        ActivationDesc,
        workspace,
        workspace_size,
        reservespace,
        reservespace_size
    );
    float *bworkspace;
    size_t bworkspace_size;

    cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        handle, 
        CUDNN_BATCHNORM_PER_ACTIVATION, 
        //CUDNN_BATCHNORM_SPATIAL, 
        CUDNN_BATCHNORM_OPS_BN_ACTIVATION, 
        //CUDNN_BATCHNORM_OPS_BN,
        //CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION, 
        dataTensor,
        outTensor, 
        outTensor,
        outTensor,
        dataTensor, 
        scalebiasTensor, 
        ActivationDesc, 
        &bworkspace_size);

    cudnnBatchNormalizationBackwardEx(
        handle,
        CUDNN_BATCHNORM_PER_ACTIVATION, 
        //CUDNN_BATCHNORM_SPATIAL, 
        CUDNN_BATCHNORM_OPS_BN_ACTIVATION, 
        //CUDNN_BATCHNORM_OPS_BN,
        //CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION,
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
        smean,
        svar,
        ActivationDesc,
        bworkspace,
        bworkspace_size,
        reservespace,
        reservespace_size
    );
    return 0;
}