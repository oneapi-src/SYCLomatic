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

    cudnnRNNDescriptor_t rnnDesc;
    cudnnRNNDataDescriptor_t xDesc;
    cudnnRNNDataDescriptor_t yDesc;
    cudnnTensorDescriptor_t hDesc;
    cudnnTensorDescriptor_t cDesc;
    cudnnDropoutDescriptor_t dropoutDesc;
    cudnnCreateRNNDescriptor(&rnnDesc);
    cudnnCreateRNNDataDescriptor(&xDesc);
    cudnnCreateRNNDataDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&hDesc);
    cudnnCreateTensorDescriptor(&cDesc);

    size_t spacesize, statesize;
    void* reservespace, *state;

    cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, hDim, hStride);
    cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, cDim, cStride);
// CHECK: auto p = (rnnDesc.set(dpct::dnnl::rnn_mode::vanilla_relu, dpct::dnnl::rnn_bias_mode::single, dir == 1 ? dpct::dnnl::rnn_direction::unidirectional : dpct::dnnl::rnn_direction::bidirectional, dpct::library_data_t::real_float, inputSize, hidenSize, projectSize, layerSize), 0);
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
// CHECK: p = (xDesc.set(CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, dpct::library_data_t::real_float, xDim[0], xDim[1], xDim[2]), 0);
    p = cudnnSetRNNDataDescriptor(xDesc, 
        CUDNN_DATA_FLOAT, 
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        xDim[0], // maxSeqLength
        xDim[1],  // batchSize
        xDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );
// CHECK: p = (yDesc.set(CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, dpct::library_data_t::real_float, yDim[0], yDim[1], yDim[2]), 0);
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
    // CHECK: handle.rnn_get_workspace_scratchpad_size(rnnDesc, dnnl::prop_kind::forward_training, xDesc, &workSpaceSize, &reserveSpaceSize);
    cudnnGetRNNTempSpaceSizes(handle, 
        rnnDesc, 
        CUDNN_FWD_MODE_TRAINING,
        xDesc, 
        &workSpaceSize,
        &reserveSpaceSize
    );

    float *xData, *yData, *hxData, *hyData, *cxData, *cyData, *weightsData, *workSpaceData, *reserveSpaceData;
    float *dxData, *dyData, *dhxData, *dhyData, *dcxData, *dcyData, *dweightsData;

    int *seqlenarray;
    // CHECK: auto e = (handle.async_rnn_forward(rnnDesc, dnnl::prop_kind::forward_training, xDesc, xData, yDesc, yData, hDesc, hxData, hyData, cDesc, cxData, cyData, weightsSpaceSize, weightsData, workSpaceSize, workSpaceData, reserveSpaceSize, reserveSpaceData), 0);
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
// CHECK: /*
// CHECK: DPCT1106:{{[0-9]+}}: Data gradient and weight gradient can't compute seperately. Replace "dpct_placeholder" with the proper argument.
// CHECK: */
// CHECK: e = (handle.async_rnn_backward(rnnDesc, yDesc, yData, dyData, xDesc, dpct_placeholder, dxData, hDesc, hxData, dhyData, dhxData, cDesc, cxData, dcyData, dcxData, weightsSpaceSize, weightsData, dpct_placeholder, workSpaceSize, workSpaceData, reserveSpaceSize, reserveSpaceData), 0);
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
// CHECK: DPCT1106:{{[0-9]+}}: Data gradient and weight gradient can't compute seperately. Replace "dpct_placeholder" with the proper argument.
// CHECK: */
// CHECK: e = (handle.async_rnn_backward(rnnDesc, yDesc, yData, dpct_placeholder, xDesc, xData, dpct_placeholder, hDesc, hxData, dpct_placeholder, dpct_placeholder, dpct_placeholder, dpct_placeholder, dpct_placeholder, weightsSpaceSize, dpct_placeholder, dweightsData, workSpaceSize, workSpaceData, reserveSpaceSize, reserveSpaceData), 0);
    e = cudnnRNNBackwardWeights_v8(
        handle,
        rnnDesc,
        CUDNN_WGRAD_MODE_ADD,
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
    
    return 0;
}