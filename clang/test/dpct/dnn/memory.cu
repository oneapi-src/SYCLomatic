// RUN: dpct -in-root %S -out-root %T/memory %S/memory.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/memory/memory.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/memory/memory.dp.cpp -o %T/memory/memory.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dnnl_utils.hpp>
// CHECK-NEXT: #include <iostream>
// CHECK-NEXT: #include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

// CHECK: template <dpct::library_data_t T>
// CHECK: struct dt_trait {
// CHECK:     typedef void type;
// CHECK: };
// CHECK: template <>
// CHECK: struct dt_trait<dpct::library_data_t::real_float> {
// CHECK:     typedef float type;
// CHECK: };
// CHECK: template <>
// CHECK: struct dt_trait<dpct::library_data_t::real_double> {
// CHECK:     typedef double type;
// CHECK: };
// CHECK: template <>
// CHECK: struct dt_trait<dpct::library_data_t::real_int32> {
// CHECK:     typedef int type;
// CHECK: };
// CHECK: template <>
// CHECK: struct dt_trait<dpct::library_data_t::real_half> {
// CHECK:     typedef float type;
// CHECK: };
template<cudnnDataType_t T>
struct dt_trait{
    typedef void type;
};
template<>
struct dt_trait<CUDNN_DATA_FLOAT>{
    typedef float type;
};
template<>
struct dt_trait<CUDNN_DATA_DOUBLE>{
    typedef double type;
};
template<>
struct dt_trait<CUDNN_DATA_INT32>{
    typedef int type;
};
template<>
struct dt_trait<CUDNN_DATA_HALF>{
    typedef float type;
};


void test() {

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor;

    cudnnCreate(&handle);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudnnSetStream(handle, stream1);

    cudnnCreateTensorDescriptor(&dataTensor);

    int on, oc, oh, ow, on_stride, oc_stride, oh_stride, ow_stride;
    size_t size;
    // CHECK: dpct::library_data_t odt;
    // CHECK: dataTensor.set(dpct::dnnl::memory_format_tag::nchw_blocked, dpct::library_data_t::real_int8_4, 1, 16, 5, 5);
    // CHECK: dataTensor.get(&odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    // CHECK: size = dataTensor.get_size();
    cudnnDataType_t odt;
    // Test 1
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 1, 16, 5, 5);
    cudnnGetTensor4dDescriptor(dataTensor, &odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    cudnnGetTensorSizeInBytes(dataTensor, &size);


    // Test 2
    // CHECK: dataTensor.set(dpct::dnnl::memory_format_tag::nhwc, dpct::library_data_t::real_float, 1, 2, 5, 5);
    // CHECK: dataTensor.get(&odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    // CHECK: size = dataTensor.get_size();
    cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 2, 5, 5);
    cudnnGetTensor4dDescriptor(dataTensor, &odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    cudnnGetTensorSizeInBytes(dataTensor, &size);


    // Test 3
    // CHECK: dataTensor.set(dpct::library_data_t::real_float, 1, 2, 5, 5, 50, 25, 5, 1);
    // CHECK: dataTensor.get(&odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    // CHECK: size = dataTensor.get_size();
    cudnnSetTensor4dDescriptorEx(dataTensor, CUDNN_DATA_FLOAT, 1, 2, 5, 5, 50, 25, 5, 1);
    cudnnGetTensor4dDescriptor(dataTensor, &odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride);
    cudnnGetTensorSizeInBytes(dataTensor, &size);


    int dims[4] = {1, 4, 5, 5};
    int odims[4] = {0, 0, 0, 0};
    int strides[4] = {100, 25, 5, 1};
    int ostrides[4] = {0, 0, 0 ,0};
    int ndims = 4, r_ndims = 4, ondims = 0;

    // Test 4
    // CHECK: dataTensor.set(dpct::library_data_t::real_float, ndims, dims, strides);
    // CHECK: dataTensor.get(r_ndims, &odt, &ondims, odims, ostrides);
    // CHECK: size = dataTensor.get_size();
    cudnnSetTensorNdDescriptor(dataTensor, CUDNN_DATA_FLOAT, ndims, dims, strides);
    cudnnGetTensorNdDescriptor(dataTensor, r_ndims, &odt, &ondims, odims, ostrides);
    cudnnGetTensorSizeInBytes(dataTensor, &size);


    // Test 5
    // CHECK: dataTensor.set(dpct::dnnl::memory_format_tag::nchw, dpct::library_data_t::real_float, ndims, dims);
    // CHECK: dataTensor.get(r_ndims, &odt, &ondims, odims, ostrides);
    // CHECK: size = dataTensor.get_size();
    cudnnSetTensorNdDescriptorEx(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ndims, dims);
    cudnnGetTensorNdDescriptor(dataTensor, r_ndims, &odt, &ondims, odims, ostrides);
    cudnnGetTensorSizeInBytes(dataTensor, &size);


    // Test 6
    dims[0] = 1;
    dims[1] = 16;
    dims[2] = 5;
    dims[3] = 5;
    // CHECK: dataTensor.set(dpct::dnnl::memory_format_tag::nchw_blocked, dpct::library_data_t::real_int8_4, ndims, dims);
    // CHECK: dataTensor.get(r_ndims, &odt, &ondims, odims, ostrides);
    // CHECK: size = dataTensor.get_size();
    cudnnSetTensorNdDescriptorEx(dataTensor, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, ndims, dims);
    cudnnGetTensorNdDescriptor(dataTensor, r_ndims, &odt, &ondims, odims, ostrides);
    cudnnGetTensorSizeInBytes(dataTensor, &size);


    // Test 7
    r_ndims = 2;
    // CHECK: dataTensor.set(dpct::dnnl::memory_format_tag::nchw, dpct::library_data_t::real_float, ndims, dims);
    // CHECK: dataTensor.get(r_ndims, &odt, &ondims, odims, ostrides);
    // CHECK: size = dataTensor.get_size();
    cudnnSetTensorNdDescriptorEx(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ndims, dims);
    cudnnGetTensorNdDescriptor(dataTensor, r_ndims, &odt, &ondims, odims, ostrides);
    cudnnGetTensorSizeInBytes(dataTensor, &size);

}

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaSetDevice(1);
    
    test();
       
    return 0;
}