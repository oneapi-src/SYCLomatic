// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test13_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test13_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test13_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test13_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test13_out

// CHECK: 13
// TEST_FEATURE: DnnlUtils_batch_normalization_forward_inference
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
    int save = 1;
    float *data, *out, *scale, *bias, *rmean, *rvar, *smean, *svar, *z;

    float alpha = 1.0f, beta = 0.f, eps = 1.f;
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

    return 0;
}