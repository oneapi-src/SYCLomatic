// RUN: dpct -out-root %T/same_func_name %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck --match-full-lines --input-file %T/same_func_name/same_func_name.dp.cpp %s

// device function
// CHECK:template <typename ValueType>
// CHECK-NEXT:int device_kernel(sycl::nd_item<3> item_ct1)
template <typename ValueType>
__device__ int device_kernel()
{
    return threadIdx.x;
}

// global function
// CHECK:template <typename ValueType>
// CHECK-NEXT:void kernel(sycl::nd_item<3> item_ct1)
template <typename ValueType>
__global__ void kernel()
{
    auto tidx = threadIdx.x;
}

// host function with different name than global function
// use template variable in global function
// CHECK:template <typename ValueType>
// CHECK-NEXT:void kernel_host(int i)
template <typename ValueType>
void kernel_host(int i)
{
    kernel<ValueType><<<16, 16>>>();
}

// host function with the same name as global function
// use template variable in global function
// CHECK:template <typename ValueType>
// CHECK-NEXT:void kernel(int i)
template <typename ValueType>
void kernel(int i)
{
    kernel<ValueType><<<16, 16>>>();
}

// host function with the same name but with different template
// use template variable in global function
// CHECK:template <typename ValueType, typename IndexType>
// CHECK-NEXT:void kernel(int i)
template <typename ValueType, typename IndexType>
void kernel(int i)
{
    kernel<ValueType><<<16, 16>>>();
}

// host function with same name and same template
// use specified template variable in global function
// CHECK:template <typename ValueType>
// CHECK-NEXT:void kernel(int i, int j)
template <typename ValueType>
void kernel(int i, int j)
{
    kernel<double><<<16, 16>>>();
}

// host function with same name but with different template
// use specified template variable in global function
// CHECK:template <typename ValueType, typename IndexType>
// CHECK-NEXT:void kernel(int i, int j)
template <typename ValueType, typename IndexType>
void kernel(int i, int j)
{
    kernel<double><<<16, 16>>>();
}

// host function with same name but without template
// use specified template variable in global function
// CHECK:void kernel(int i, int j)
void kernel(int i, int j)
{
    kernel<double><<<16, 16>>>();
}


// CHECK:template <typename ValueType>
// CHECK-NEXT:void kernel_2(sycl::nd_item<3> item_ct1)
// CHECK-NEXT:{
// CHECK-NEXT:    auto tidx = item_ct1.get_local_id(2);
// CHECK-NEXT:}
template <typename ValueType>
__global__ void kernel_2()
{
    auto tidx = threadIdx.x;
}

// CHECK:template <typename ValueType>
// CHECK-NEXT:void kernel_2(int a,
// CHECK-NEXT:              sycl::nd_item<3> item_ct1)
// CHECK-NEXT:{
// CHECK-NEXT:    auto tidx = item_ct1.get_local_id(2);
// CHECK-NEXT:}
template <typename ValueType>
__global__ void kernel_2(int a)
{
    auto tidx = threadIdx.x;
}
