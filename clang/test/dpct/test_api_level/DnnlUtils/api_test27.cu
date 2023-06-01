// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test27_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test27_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test27_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test27_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test27_out

// CHECK: 38
// TEST_FEATURE: DnnlUtils_async_dropout_backward
// TEST_FEATURE: DnnlUtils_async_dropout_forward
// TEST_FEATURE: DnnlUtils_dropout_desc
// TEST_FEATURE: DnnlUtils_get_dropout_state_size
// TEST_FEATURE: DnnlUtils_get_dropout_workspace_size

#include<cuda_runtime.h>
#include<cudnn.h>
#include<iostream>

int main(){

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);

    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;
    float *data, *out, *d_data, *d_out, *out2;

    size_t state_size;
    void *state, *state2;
    cudnnDropoutGetStatesSize(handle, &state_size);

    cudaMallocManaged(&data, ele_num * sizeof(float));
    cudaMallocManaged(&out, ele_num * sizeof(float));
    cudaMallocManaged(&d_data, ele_num * sizeof(float));
    cudaMallocManaged(&d_out, ele_num * sizeof(float));

    for(int i = 0; i < ele_num; i++){
      data[i] = 10;
      d_out[i] = 20;
    }

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    float dropout = 0.5f;

    size_t reserve_size;
    void *reserve;

    cudnnDropoutDescriptor_t desc, desc2;
    cudnnCreateDropoutDescriptor(&desc);

    cudaMallocManaged(&state, state_size);


    cudnnSetDropoutDescriptor(desc, handle, dropout, state, state_size, 1231);

    float d;
    void *st;
    unsigned long long se;
    cudnnGetDropoutDescriptor(desc, handle, &d, &sd, &se);

    cudnnDropoutGetReserveSpaceSize(dataTensor, &reserve_size);
    cudaMalloc(&reserve, reserve_size);

    cudnnRestoreDropoutDescriptor(desc2, handle, dropout, state, state_size, 1231);

    cudnnDropoutForward(handle, desc, dataTensor, data, outTensor, out, reserve, reserve_size);

    cudaDeviceSynchronize();
    for(int i = 0; i < ele_num; i++){
      std::cout << out[i] << " ";
      if((i + 1)%5 == 0) {
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;

    cudnnDropoutBackward(handle, desc, dataTensor, d_out, outTensor, d_data, reserve, reserve_size);
    cudaDeviceSynchronize();
    cudnnDestroyDropoutDescriptor(desc);
    for(int i = 0; i < ele_num; i++){
      std::cout << d_data[i] << " ";
      if((i + 1)%5 == 0) {
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
    return 0;

}