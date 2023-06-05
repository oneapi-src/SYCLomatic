// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/rnn %S/rnn.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/rnn/rnn.dp.cpp --match-full-lines %s

// CHECK: #include <dpct/dnnl_utils.hpp>
// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include <iostream>
// CHECK: #include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    int hidenSize = 2;
    int layerSize = 3;
    int inputSize = 2;
    int projectSize = 2;
    int batchSize = 3;
    int maxSeqLength = 4;
    int dir = 2;

    int hDim[3] = {dir * layerSize, batchSize, projectSize};
    int hStride[3] = {hDim[1] * hDim[2], hDim[2], 1};

    int cDim[3] = {layerSize * dir, batchSize, hidenSize};
    int cStride[3] = {cDim[2] * cDim[1], cDim[2], 1};
    int xDim[3] = {maxSeqLength, batchSize, inputSize};
    int yDim[3] = {maxSeqLength, batchSize, dir * projectSize};

    int h_size = hDim[0] * hDim[1] * hDim[2];
    int c_size = cDim[0] * cDim[1] * cDim[2];
    int x_size = xDim[0] * xDim[1] * xDim[2];
    int y_size = yDim[0] * yDim[1] * yDim[2];

    cudnnHandle_t handle;

    cudnnCreate(&handle);
// CHECK:    dpct::dnnl::rnn_desc rnnDesc;
// CHECK:    dpct::dnnl::memory_desc_ext xDesc;
// CHECK:    dpct::dnnl::memory_desc_ext yDesc;
// CHECK:    dpct::dnnl::memory_desc_ext hDesc;
// CHECK:    dpct::dnnl::memory_desc_ext cDesc;
    cudnnRNNDescriptor_t rnnDesc;
    cudnnRNNDataDescriptor_t xDesc;
    cudnnRNNDataDescriptor_t yDesc;
    cudnnTensorDescriptor_t hDesc;
    cudnnTensorDescriptor_t cDesc;
    cudnnCreateRNNDescriptor(&rnnDesc);
    cudnnCreateRNNDataDescriptor(&xDesc);
    cudnnCreateRNNDataDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&hDesc);
    cudnnCreateTensorDescriptor(&cDesc);

    size_t spacesize, statesize;
    void* reservespace, *state;

    cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, hDim, hStride);
    cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, cDim, cStride);
// CHECK: auto p = DPCT_CHECK_ERROR(rnnDesc.set(dpct::dnnl::rnn_mode::vanilla_relu, dpct::dnnl::rnn_bias_mode::single, dir == 1 ? dpct::dnnl::rnn_direction::unidirectional : dpct::dnnl::rnn_direction::bidirectional, dpct::library_data_t::real_float, inputSize, hidenSize, projectSize, layerSize));
    auto p = cudnnSetRNNDescriptor_v8(rnnDesc,
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_RNN_RELU,
        CUDNN_RNN_SINGLE_INP_BIAS,
        dir == 1 ? CUDNN_UNIDIRECTIONAL : CUDNN_BIDIRECTIONAL,
        CUDNN_LINEAR_INPUT,
        CUDNN_DATA_FLOAT,
        CUDNN_DATA_FLOAT,
        CUDNN_DEFAULT_MATH,
        inputSize,  // inputSize
        hidenSize,  // hiddenSize
        projectSize,  // projSize
        layerSize,  // numLayers
        NULL,
        CUDNN_RNN_PADDED_IO_ENABLED
    );

    int seqLenArray[3];
    seqLenArray[0] = maxSeqLength;
    seqLenArray[1] = maxSeqLength;
    seqLenArray[2] = maxSeqLength;
// CHECK: p = DPCT_CHECK_ERROR(xDesc.set(dpct::dnnl::rnn_memory_format_tag::tnc, dpct::library_data_t::real_float, xDim[0], xDim[1], xDim[2]));
    p = cudnnSetRNNDataDescriptor(xDesc, 
        CUDNN_DATA_FLOAT, 
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        xDim[0], // maxSeqLength
        xDim[1],  // batchSize
        xDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );
// CHECK: p = DPCT_CHECK_ERROR(yDesc.set(dpct::dnnl::rnn_memory_format_tag::tnc, dpct::library_data_t::real_float, yDim[0], yDim[1], yDim[2]));
    p = cudnnSetRNNDataDescriptor(yDesc,
        CUDNN_DATA_FLOAT,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        yDim[0], // maxSeqLength
        yDim[1],  // batchSize
        yDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );

    size_t weightsSpaceSize, workSpaceSize, reserveSpaceSize;
    // CHECK: handle.rnn_get_weight_space_size(rnnDesc, &weightsSpaceSize);
    cudnnGetRNNWeightSpaceSize(handle, rnnDesc, &weightsSpaceSize);
    // CHECK: handle.rnn_get_scratchpad_workspace_size(rnnDesc, dnnl::prop_kind::forward_training, xDesc, &workSpaceSize, &reserveSpaceSize);
    cudnnGetRNNTempSpaceSizes(handle, 
        rnnDesc, 
        CUDNN_FWD_MODE_TRAINING,
        xDesc, 
        &workSpaceSize,
        &reserveSpaceSize
    );

    float *xData, *yData, *hxData, *hyData, *cxData, *cyData, *weightsData, *workSpaceData, *reserveSpaceData;
    float *dxData, *dyData, *dhxData, *dhyData, *dcxData, *dcyData, *dweightsData;
    cudnnWgradMode_t WGMode = CUDNN_WGRAD_MODE_ADD;
    int *seqlenarray;
    // CHECK: auto e = DPCT_CHECK_ERROR(handle.async_rnn_forward(rnnDesc, dnnl::prop_kind::forward_training, xDesc, xData, yDesc, yData, hDesc, hxData, hyData, cDesc, cxData, cyData, weightsSpaceSize, weightsData, workSpaceSize, workSpaceData, reserveSpaceSize, reserveSpaceData));
    auto e = cudnnRNNForward(
        handle, 
        rnnDesc, 
        CUDNN_FWD_MODE_TRAINING, 
        seqlenarray, 
        xDesc, 
        xData, 
        yDesc, 
        yData, 
        hDesc, 
        hxData, 
        hyData, 
        cDesc, 
        cxData,
        cyData,
        weightsSpaceSize, 
        weightsData, 
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );
// CHECK: e = DPCT_CHECK_ERROR(handle.async_rnn_backward(rnnDesc, yDesc, yData, dyData, xDesc, xData, dxData, hDesc, hxData, dhyData, dhxData, cDesc, cxData, dcyData, dcxData, weightsSpaceSize, weightsData, dweightsData, workSpaceSize, workSpaceData, reserveSpaceSize, reserveSpaceData));
    e = cudnnRNNBackwardData_v8(
        handle,
        rnnDesc,
        seqlenarray,
        yDesc,
        yData,
        dyData,
        xDesc,
        dxData,
        hDesc,
        hxData,
        dhyData,
        dhxData,
        cDesc,
        cxData,
        dcyData,
        dcxData,
        weightsSpaceSize,
        weightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );
// CHECK: /*
// CHECK: DPCT1027:{{[0-9]+}}: The call to cudnnRNNBackwardWeights_v8 was replaced with 0 because this call and cudnnRNNBackwardData_v8 are migrated to a single function call async_rnn_backward
// CHECK: */
// CHECK: e = 0;
    e = cudnnRNNBackwardWeights_v8(
        handle,
        rnnDesc,
        WGMode,
        seqlenarray,
        xDesc,
        xData,
        hDesc,
        hxData,
        yDesc,
        yData,
        weightsSpaceSize,
        dweightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

   float *y1Data;
// CHECK:     /*
// CHECK:     DPCT1007:{{[0-9]+}}: Migration of cudnnRNNBackwardData_v8 is not supported.
// CHECK:     */
    cudnnRNNBackwardData_v8(
        handle,
        rnnDesc,
        seqlenarray,
        yDesc,
        y1Data,
        dyData,
        xDesc,
        dxData,
        hDesc,
        hxData,
        dhyData,
        dhxData,
        cDesc,
        cxData,
        dcyData,
        dcxData,
        weightsSpaceSize,
        weightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );
    
    cudaMalloc(&y1Data, 100);
// CHECK:     /*
// CHECK:     DPCT1007:{{[0-9]+}}: Migration of cudnnRNNBackwardWeights_v8 is not supported.
// CHECK:     */
    cudnnRNNBackwardWeights_v8(
        handle,
        rnnDesc,
        WGMode,
        seqlenarray,
        xDesc,
        xData,
        hDesc,
        hxData,
        yDesc,
        y1Data,
        weightsSpaceSize,
        dweightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

    float *y2Data;
// CHECK:     /*
// CHECK:     DPCT1007:{{[0-9]+}}: Migration of cudnnRNNBackwardData_v8 is not supported.
// CHECK:     */
    cudnnRNNBackwardData_v8(
        handle,
        rnnDesc,
        seqlenarray,
        yDesc,
        y2Data,
        dyData,
        xDesc,
        dxData,
        hDesc,
        hxData,
        dhyData,
        dhxData,
        cDesc,
        cxData,
        dcyData,
        dcxData,
        weightsSpaceSize,
        weightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

    float *y3Data;
// CHECK:     /*
// CHECK:     DPCT1007:{{[0-9]+}}: Migration of cudnnRNNBackwardWeights_v8 is not supported.
// CHECK:     */
    e = cudnnRNNBackwardWeights_v8(
        handle,
        rnnDesc,
        WGMode,
        seqlenarray,
        xDesc,
        xData,
        hDesc,
        hxData,
        yDesc,
        y3Data,
        weightsSpaceSize,
        dweightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

    float *y4Data;
    bool flag;
// CHECK:     /*
// CHECK:     DPCT1007:{{[0-9]+}}: Migration of cudnnRNNBackwardData_v8 is not supported.
// CHECK:     */
    if(flag)    
      cudnnRNNBackwardData_v8(
        handle,
        rnnDesc,
        seqlenarray,
        yDesc,
        y4Data,
        dyData,
        xDesc,
        dxData,
        hDesc,
        hxData,
        dhyData,
        dhxData,
        cDesc,
        cxData,
        dcyData,
        dcxData,
        weightsSpaceSize,
        weightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

// CHECK:     /*
// CHECK:     DPCT1007:{{[0-9]+}}: Migration of cudnnRNNBackwardWeights_v8 is not supported.
// CHECK:     */
    e = cudnnRNNBackwardWeights_v8(
        handle,
        rnnDesc,
        WGMode,
        seqlenarray,
        xDesc,
        xData,
        hDesc,
        hxData,
        yDesc,
        y4Data,
        weightsSpaceSize,
        dweightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );


    return 0;
}