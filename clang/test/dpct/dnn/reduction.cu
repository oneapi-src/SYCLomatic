// RUN: dpct -in-root %S -out-root %T/reduction %S/reduction.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/reduction/reduction.dp.cpp --match-full-lines %s

// CHECK: #include <dpct/dnnl_utils.hpp>
// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include <iostream>
// CHECK: #include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#define DT float
int main() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&outTensor);

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 2, oh = 6, ow = 1;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw);
    cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow);

    DT *data, *out;
    std::vector<DT> host_data(in * ic * iw * ih, 0);
    std::vector<DT> host_out(on * oc * ow * oh, 0);

    for(int i = 0; i < in * ic * ih * iw; i++) {
        host_data[i] = i - 25.f;
    }
    host_data[10] = 0.f;
    for(int i = 0; i < on * oc * oh * ow; i++) {
        host_out[i] = i;
    }

    cudaMalloc(&data, sizeof(DT) * in * ic * ih * iw);
    cudaMalloc(&out, sizeof(DT) * on * oc * oh * ow);

    cudaMemcpy(data, host_data.data(), sizeof(DT) * in * ic * ih * iw, cudaMemcpyHostToDevice);
    cudaMemcpy(out, host_out.data(), sizeof(DT) * on * oc * oh * ow, cudaMemcpyHostToDevice);

    float alpha = 2.5f, beta = 1.5f;
    // CHECK: dpct::dnnl::reduction_op reducedesc;
    cudnnReduceTensorDescriptor_t reducedesc;
    cudnnCreateReduceTensorDescriptor(&reducedesc);
    // CHECK: reducedesc = dpct::dnnl::reduction_op::mul_no_zeros;
    cudnnSetReduceTensorDescriptor(
        reducedesc,
        CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES);

    void *ws;
    size_t ws_size;
    // CHECK: ws_size = 0;
    cudnnGetReductionWorkspaceSize(
        handle,
        reducedesc,
        dataTensor,
        outTensor,
        &ws_size);

    cudaMalloc(&ws, ws_size);
    // CHECK: /*
    // CHECK: DPCT1007:{{[0-9]+}}: Migration of reduction index is not supported.
    // CHECK: */
    // CHECK: handle.async_reduction(reducedesc, alpha, dataTensor, data, beta, outTensor, out);
    cudnnReduceTensor(
        handle,
        reducedesc,
        0,
        0,
        ws,
        ws_size,
        &alpha,
        dataTensor,
        data,
        &beta,
        outTensor,
        out);

    cudaMemcpy(host_out.data(), out, sizeof(DT) * on * oc * oh * ow, cudaMemcpyDeviceToHost);
    cudnnDestroy(handle);
    cudaFree(data);
    cudaFree(out);
    return 0;
}
