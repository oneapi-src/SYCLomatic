// RUN: dpct --format-range=none --use-custom-helper=api -out-root %T/DnnlUtils/api_test2_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/DnnlUtils/api_test2_out/MainSourceFiles.yaml | wc -l > %T/DnnlUtils/api_test2_out/count.txt
// RUN: FileCheck --input-file %T/DnnlUtils/api_test2_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DnnlUtils/api_test2_out

// CHECK: 10

#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
// TEST_FEATURE: DnnlUtils_memory_desc_ext
// TEST_FEATURE: LibCommonUtils_library_data_t
// TEST_FEATURE: DnnlUtils_memory_format_tag

int main() {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor;

    cudnnCreate(&handle);

    cudnnCreateTensorDescriptor(&dataTensor);

    int on, oc, oh, ow, on_stride, oc_stride, oh_stride, ow_stride;
    size_t size;
    cudnnDataType_t odt;

    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 1, 16, 5, 5);
    cudnnGetTensor4dDescriptor(dataTensor, &odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    cudnnGetTensorSizeInBytes(dataTensor, &size);
    return 0;
}
