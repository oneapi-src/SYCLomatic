// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test10_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test10_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test10_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test10_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test10_out

// CHECK: 38

#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
// TEST_FEATURE: DnnlUtils_softmax_algorithm
// TEST_FEATURE: DnnlUtils_softmax_mode
// TEST_FEATURE: DnnlUtils_softmax_forward
// TEST_FEATURE: DnnlUtils_softmax_backward

int main() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, diffdataTensor, diffoutTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);
    cudnnCreateTensorDescriptor(&diffdataTensor);
    cudnnCreateTensorDescriptor(&diffoutTensor);
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    //using float = dt_trait<T>::type;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    float *data, *out, *diffdata, *diffout;
    std::vector<float> host_data(ele_num);
    std::vector<float> host_out(ele_num);
    std::vector<float> host_diffdata(ele_num);
    std::vector<float> host_diffout(ele_num);
    for(int i = 0; i < ele_num; i++) {
        host_data[i] = i * 0.1f;
        host_out[i] = i;
        host_diffdata[i] = i;
        host_diffout[i] = 1.f;
    }

    cudaMalloc(&data, ele_num * sizeof(float));
    cudaMalloc(&out, ele_num * sizeof(float));
    cudaMalloc(&diffdata, ele_num * sizeof(float));
    cudaMalloc(&diffout, ele_num * sizeof(float));

    cudaMemcpy(data, host_data.data(), ele_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), ele_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(diffdata, host_diffdata.data(), ele_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(diffout, host_diffout.data(), ele_num * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.5f, beta = 0.f;
    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, dataTensor, data, &beta, outTensor, out);
    cudaMemcpy(host_out.data(), out, ele_num * sizeof(float), cudaMemcpyDeviceToHost);
    alpha = 2.f, beta = 0.f;
    cudaDeviceSynchronize();
    cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, outTensor, out, diffoutTensor, diffout, &beta, diffdataTensor, diffdata);
    cudaDeviceSynchronize();

    cudaMemcpy(host_diffdata.data(), diffdata, ele_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    cudaFree(diffdata);
    cudaFree(diffout);
    return 0;
}
