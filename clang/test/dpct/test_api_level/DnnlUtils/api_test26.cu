// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test26_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test26_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test26_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test26_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test26_out

// CHECK: 32
// TEST_FEATURE: DnnlUtils_rnn_mode
// TEST_FEATURE: DnnlUtils_rnn_bias_mode
// TEST_FEATURE: DnnlUtils_rnn_direction
// TEST_FEATURE: DnnlUtils_async_rnn_forward
// TEST_FEATURE: DnnlUtils_async_rnn_backward
// TEST_FEATURE: DnnlUtils_rnn_get_scratchpad_workspace_size
// TEST_FEATURE: DnnlUtils_rnn_get_weight_space_size
// TEST_FEATURE: DnnlUtils_rnn_memory_format_tag
// TEST_FEATURE: DnnlUtils_rnn_desc

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

    cudnnRNNMode_t mode = CUDNN_RNN_RELU;
    cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_SINGLE_INP_BIAS;
    cudnnDirectionMode_t rdir = CUDNN_UNIDIRECTIONAL;
    auto p = cudnnSetRNNDescriptor_v8(rnnDesc,
        CUDNN_RNN_ALGO_STANDARD,
        mode,
        bias_mode,
        rdir,
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

    p = cudnnSetRNNDataDescriptor(xDesc, 
        CUDNN_DATA_FLOAT, 
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        xDim[0], // maxSeqLength
        xDim[1],  // batchSize
        xDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );

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

    cudnnGetRNNWeightSpaceSize(handle, rnnDesc, &weightsSpaceSize);

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