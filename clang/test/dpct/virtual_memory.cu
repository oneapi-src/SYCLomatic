// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --use-experimental-features=virtual_mem -out-root %T/virtual_memory %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/virtual_memory/virtual_memory.dp.cpp
#include <cuda.h>
#include <iostream>

#define SIZE 100

int main() {
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext context;
    cuCtxCreate(&context, 0, device);

// CHECK:  dpct::experimental::mem_prop prop = {};
// CHECK:  prop.type = 0;
// CHECK:  prop.location.type = 1;
// CHECK:  prop.location.id = device;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    size_t granularity;
// CHECK:    granularity = sycl::ext::oneapi::experimental::get_mem_granularity(dpct::get_device(prop.location.id), dpct::get_device(prop.location.id).get_context(), sycl::ext::oneapi::experimental::granularity_mode::minimum);
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);   
    size_t POOL_SIZE =  granularity;

// CHECK:    dpct::device_ptr reserved_addr;
// CHECK:    dpct::experimental::physical_mem_ptr allocHandle;
// CHECK:    reserved_addr = (dpct::device_ptr)sycl::ext::oneapi::experimental::reserve_virtual_mem((uintptr_t)0, POOL_SIZE, dpct::get_current_device().get_context());
// CHECK:    allocHandle = new sycl::ext::oneapi::experimental::physical_mem(dpct::get_device(prop.location.id), dpct::get_device(prop.location.id).get_context(), POOL_SIZE);
// CHECK:    allocHandle->map((uintptr_t)reserved_addr, POOL_SIZE, sycl::ext::oneapi::experimental::address_access_mode::read_write, 0);
    CUdeviceptr reserved_addr;
    CUmemGenericAllocationHandle allocHandle;
    cuMemAddressReserve(&reserved_addr, POOL_SIZE, 0, 0, 0);
    cuMemCreate(&allocHandle, POOL_SIZE, &prop, 0);
    cuMemMap(reserved_addr, POOL_SIZE, 0, allocHandle, 0);

    // CHECK: prop.location.id = dpct::get_device_id(allocHandle->get_device());
    cuMemGetAllocationPropertiesFromHandle(&prop, allocHandle);

// CHECK:  dpct::experimental::mem_access_desc accessDesc = {};
// CHECK:  accessDesc.location.type = 1;
// CHECK:  accessDesc.location.id = prop.location.id;
// CHECK:  accessDesc.flags = sycl::ext::oneapi::experimental::address_access_mode::read_write;
// CHECK:  sycl::ext::oneapi::experimental::set_access_mode(reserved_addr, POOL_SIZE, accessDesc.flags, dpct::get_device(accessDesc.location.id).get_context());
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = prop.location.id;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(reserved_addr, POOL_SIZE, &accessDesc, 1);
    int* host_data = new int[SIZE];
    int* host_data2 = new int[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        host_data[i] = i;
        host_data2[i] = 0;
    }

    cuMemcpyHtoD(reserved_addr, host_data, SIZE * sizeof(int));
    cuMemcpyDtoH(host_data2, reserved_addr, SIZE * sizeof(int));

    for (int i = 0; i < SIZE; ++i) {
        if(host_data[i] != host_data2[i]) {
          std::cout << "test failed" << std::endl;
          exit(-1);
        }
    }
    std::cout << "test passed" << std::endl;

    // CHECK:  sycl::ext::oneapi::experimental::unmap(reserved_addr, POOL_SIZE, dpct::get_current_device().get_context());
    // CHECK:  delete (allocHandle);
    // CHECK:  sycl::ext::oneapi::experimental::free_virtual_mem((uintptr_t)reserved_addr, POOL_SIZE, dpct::get_current_device().get_context());
    cuMemUnmap(reserved_addr, POOL_SIZE);
    cuMemRelease(allocHandle);
    cuMemAddressFree(reserved_addr, POOL_SIZE);

    delete[] host_data;
    delete[] host_data2;

    cuCtxDestroy(context);
    return 0;
}
