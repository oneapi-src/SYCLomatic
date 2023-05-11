// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test8_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test8_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test8_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test8_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test8_out

// CHECK: 54

#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
// TEST_FEATURE: DnnlUtils_pooling_desc
// TEST_FEATURE: DnnlUtils_pooling_forward
// TEST_FEATURE: DnnlUtils_pooling_backward

int main() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, diffdataTensor, diffoutTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnPoolingDescriptor_t desc;
    cudnnCreatePoolingDescriptor(&desc);
    cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 4, 4, 3, 3, 2, 2);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    //cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);
    cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    //cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w);

    int on, oc, oh, ow;
    cudnnGetPooling2dForwardOutputDim(desc, dataTensor, &on, &oc, &oh, &ow);

    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);
    int ele_num2 = on * oc * oh * ow;

    float *data, *out, *diffdata, *diffout;
    std::vector<float> host_data(ele_num);
    std::vector<float> host_out(ele_num2);
    std::vector<float> host_diffdata(ele_num);
    std::vector<float> host_diffout(ele_num2);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i * 0.1f;
        //host_out[i] = i;
        host_diffdata[i] = i;
        //host_diffout[i] = 1.f;
    }
    for(int i = 0; i < ele_num2; i++) {
        //host_data[i] = i * 0.1f;
        host_out[i] = i;
        //host_diffdata[i] = i;
        host_diffout[i] = 1.f;
    }

    cudaMalloc(&data, ele_num * sizeof(float));
    cudaMalloc(&out, ele_num2 * sizeof(float));
    cudaMalloc(&diffdata, ele_num * sizeof(float));
    cudaMalloc(&diffout, ele_num2 * sizeof(float));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), ele_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), ele_num2 * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.5f, beta = 1.f;
    cudnnPoolingForward(handle, desc, &alpha, dataTensor, data, &beta, outTensor, out);
    cudaMemcpy(host_out.data(), out, ele_num2 * sizeof(float), cudaMemcpyDeviceToHost);
    alpha = 1.5f, beta = 1.f;
    cudaDeviceSynchronize();
    cudnnPoolingBackward(handle, desc, &alpha, outTensor, out, diffoutTensor, diffout, dataTensor, data, &beta, diffdataTensor, diffdata);
    cudaDeviceSynchronize();
    //check(s);
    cudaMemcpy(host_diffdata.data(), diffdata, ele_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    cudaFree(diffdata);
    cudaFree(diffout);
    return 0;
}
