// RUN: dpct --format-range=none --use-experimental-features=virtual_memory -out-root %T/virtual_memory %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
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

// CHECK:    dpct::experimental::mem_prop prop = {};
// CHECK:    prop.type = dpct::experimental::mem_allocation_type::MEM_ALLOCATION_TYPE_DEFAULT;
// CHECK:    prop.location.type = dpct::experimental::mem_location_type::MEM_LOCATION_TYPE_DEVICE;
// CHECK:    prop.location.id = device;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    size_t granularity;
// CHECK:    dpct::experimental::mem_get_allocation_granularity(&granularity, &prop, dpct::experimental::granularity_flags::GRANULARITY_FLAGS_MINIMUM);
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    size_t POOL_SIZE =  granularity;

// CHECK:    dpct::device_ptr reserved_addr;
// CHECK:    dpct::experimental::mem_handle allocHandle;
// CHECK:    dpct::experimental::mem_address_reserve(&reserved_addr, POOL_SIZE, 0, 0, 0);
// CHECK:    dpct::experimental::mem_create(&allocHandle, POOL_SIZE, &prop, 0);
// CHECK:    dpct::experimental::mem_map(reserved_addr, POOL_SIZE, 0, allocHandle, 0);
    CUdeviceptr reserved_addr;
    CUmemGenericAllocationHandle allocHandle;
    cuMemAddressReserve(&reserved_addr, POOL_SIZE, 0, 0, 0);
    cuMemCreate(&allocHandle, POOL_SIZE, &prop, 0);
    cuMemMap(reserved_addr, POOL_SIZE, 0, allocHandle, 0);

// CHECK:    dpct::experimental::mem_access_desc accessDesc = {};
// CHECK:    accessDesc.location.type = dpct::experimental::mem_location_type::MEM_LOCATION_TYPE_DEVICE;
// CHECK:    accessDesc.location.id = device;
// CHECK:    accessDesc.flags = dpct::experimental::address_access_flags::ADDRESS_ACCESS_FLAGS_READ_WRITE;
// CHECK:    dpct::experimental::mem_set_access(reserved_addr, POOL_SIZE, &accessDesc, 1);
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
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

// CHECK:    dpct::experimental::mem_unmap(reserved_addr, POOL_SIZE);
// CHECK:    dpct::experimental::mem_release(allocHandle);
// CHECK:    dpct::experimental::mem_address_free(reserved_addr, POOL_SIZE);
    cuMemUnmap(reserved_addr, POOL_SIZE);
    cuMemRelease(allocHandle);
    cuMemAddressFree(reserved_addr, POOL_SIZE);

    delete[] host_data;
    delete[] host_data2;

    cuCtxDestroy(context);
    return 0;
}
