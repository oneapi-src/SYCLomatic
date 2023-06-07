// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test16_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test16_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test16_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test16_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test16_out

// CHECK: 18
// TEST_FEATURE: DnnlUtils_batch_normalization_backward
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
    int sbn = 1, sbc = 4, sbh = 5, sbw = 5;
    int ele_num = in* ic * ih * iw;
    int oele_num = on* oc * oh * ow;
    int sele_num = sbn*sbc * sbh * sbw;
    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;
    float *diffout, *diffdata, *diffscale, *diffbias;
    float alpha = 1.f, beta = 0.f, eps = 1.f;
    double factor = 0.1f;
    auto status = cudnnBatchNormalizationForwardTraining(
        handle,
        CUDNN_BATCHNORM_PER_ACTIVATION,
        //CUDNN_BATCHNORM_SPATIAL,
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

    return 0;
}