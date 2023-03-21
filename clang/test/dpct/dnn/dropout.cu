// RUN: dpct -in-root %S -out-root %T/dropout %S/dropout.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/dropout/dropout.dp.cpp --match-full-lines %s

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
    // CHECK: state_size = handle.get_dropout_state_size();
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
    // CHECK: dpct::dnnl::dropout_desc desc, desc2;
    cudnnDropoutDescriptor_t desc, desc2;
    // CHECK: desc.init();
    cudnnCreateDropoutDescriptor(&desc);

    cudaMallocManaged(&state, state_size);


    cudnnSetDropoutDescriptor(desc, handle, dropout, state, state_size, 1231);

    float d;
    void *st;
    unsigned long long se;
    // CHECK: desc.get(&d, &st, &se);
    cudnnGetDropoutDescriptor(desc, handle, &d, &st, &se);
    // CHECK: reserve_size = dpct::dnnl::engine_ext::get_dropout_workspace_size(dataTensor);
    cudnnDropoutGetReserveSpaceSize(dataTensor, &reserve_size);
    cudaMalloc(&reserve, reserve_size);

    // CHECK: desc2.restore(handle, dropout, state, state_size, 1231);
    cudnnRestoreDropoutDescriptor(desc2, handle, dropout, state, state_size, 1231);

    // CHECK: handle.async_dropout_forward(desc, dataTensor, data, outTensor, out, reserve, reserve_size);
    cudnnDropoutForward(handle, desc, dataTensor, data, outTensor, out, reserve, reserve_size);

    cudaDeviceSynchronize();
    for(int i = 0; i < ele_num; i++){
      std::cout << out[i] << " ";
      if((i + 1)%5 == 0) {
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
    // CHECK: handle.async_dropout_backward(desc, dataTensor, d_out, outTensor, d_data, reserve, reserve_size);
    cudnnDropoutBackward(handle, desc, dataTensor, d_out, outTensor, d_data, reserve, reserve_size);
    cudaDeviceSynchronize();

    // CHECK: /*
    // CHECK: DPCT1026:{{[0-9]+}}: The call to cudnnDestroyDropoutDescriptor was removed because this call is redundant in SYCL.
    // CHECK: */
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
