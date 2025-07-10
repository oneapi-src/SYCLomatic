// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1
// UNSUPPORTED: v8.0, v9.0, v9.1

/// Device Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetUuid | FileCheck %s -check-prefix=CUDEVICEGETUUID
// CUDEVICEGETUUID: CUDA API:
// CUDEVICEGETUUID-NEXT:   cuDeviceGetUuid(pu /*CUuuid **/, d /*CUdevice*/);
// CUDEVICEGETUUID-NEXT: Is migrated to:
// CUDEVICEGETUUID-NEXT:   *pu = dpct::get_device(d).get_device_info().get_uuid();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAddressReserve | FileCheck %s -check-prefix=CUMEMADDRESSRESERVE
// CUMEMADDRESSRESERVE: CUDA API:
// CUMEMADDRESSRESERVE-NEXT:   cuMemAddressReserve(ptr /*CUdeviceptr **/, size /*size_t*/,
// CUMEMADDRESSRESERVE-NEXT:                      alignment /*size_t*/, addr /*CUdeviceptr*/,
// CUMEMADDRESSRESERVE-NEXT:                      flags /*unsigned long long*/);
// CUMEMADDRESSRESERVE-NEXT: Is migrated to (with the option --use-experimental-features=virtual_mem):
// CUMEMADDRESSRESERVE-NEXT:   *ptr = (dpct::device_ptr)sycl::ext::oneapi::experimental::reserve_virtual_mem((uintptr_t)addr, size, dpct::get_current_device().get_context());


// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemGetAllocationPropertiesFromHandle | FileCheck %s -check-prefix=CUMEMGETALLOCATIONPROPERTIESFROMHANDLE
// CUMEMGETALLOCATIONPROPERTIESFROMHANDLE: CUDA API:
// CUMEMGETALLOCATIONPROPERTIESFROMHANDLE-NEXT:   cuMemGetAllocationPropertiesFromHandle(&prop/*CUmemAllocationProp **/, allocHandle/*CUmemGenericAllocationHandle*/);
// CUMEMGETALLOCATIONPROPERTIESFROMHANDLE-NEXT: Is migrated to (with the option --use-experimental-features=virtual_mem):
// CUMEMGETALLOCATIONPROPERTIESFROMHANDLE-NEXT:    prop.location.id = dpct::get_device_id(allocHandle->get_device());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuPointerGetAttribute | FileCheck %s -check-prefix=CUPOINTERGETATTRIBUTE
// CUPOINTERGETATTRIBUTE: CUDA API:
// CUPOINTERGETATTRIBUTE-NEXT:   cuPointerGetAttribute(base_ptr/*void **/, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR/*CUpointer_attribute*/, (CUdeviceptr)ptr/*CUdeviceptr*/);
// CUPOINTERGETATTRIBUTE-NEXT: Is migrated to (with the option --usm-level=none):
// CUPOINTERGETATTRIBUTE-NEXT:   base_ptr = dpct::get_base_addr((dpct::device_ptr)ptr);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamGetCtx | FileCheck %s -check-prefix=CUSTREAMGETCTX
// CUSTREAMGETCTX: CUDA API:
// CUSTREAMGETCTX-NEXT:   cuStreamGetCtx(stream/*CUstream*/, &context/*CUcontext **/);
// CUSTREAMGETCTX-NEXT: Is migrated to:
// CUSTREAMGETCTX-NEXT:   context = dpct::get_device_id(stream->get_device());
