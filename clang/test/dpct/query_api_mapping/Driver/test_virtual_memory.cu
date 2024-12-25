// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
/// Virtual Memory Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemMap | FileCheck %s -check-prefix=cuMemMap
// cuMemMap:  CUDA API:
// cuMemMap-NEXT:    CUmemGenericAllocationHandle handle;
// cuMemMap-NEXT:    cuMemMap(ptr /*CUdeviceptr*/, size /*size_t*/, offset /*size_t*/,
// cuMemMap-NEXT:             handle /*CUmemGenericAllocationHandle*/,
// cuMemMap-NEXT:             flags /*unsigned long long */);
// cuMemMap-NEXT:  Is migrated to (with the option --use-experimental-features=virtual_mem):
// cuMemMap-NEXT:    dpct::experimental::physical_mem_ptr handle;
// cuMemMap-NEXT:    handle->map((uintptr_t)ptr, size, sycl::ext::oneapi::experimental::address_access_mode::read_write, offset);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemRelease | FileCheck %s -check-prefix=cuMemRelease
// cuMemRelease: CUDA API:
// cuMemRelease-NEXT:   CUmemGenericAllocationHandle handle;
// cuMemRelease-NEXT:   cuMemRelease(handle /*CUmemGenericAllocationHandle */);
// cuMemRelease-NEXT: Is migrated to (with the option --use-experimental-features=virtual_mem):
// cuMemRelease-NEXT:   dpct::experimental::physical_mem_ptr handle;
// cuMemRelease-NEXT:   delete (handle);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemUnmap | FileCheck %s -check-prefix=cuMemUnmap
// cuMemUnmap: CUDA API:
// cuMemUnmap-NEXT:   cuMemUnmap(ptr /*CUdeviceptr*/, size /*size_t*/);
// cuMemUnmap-NEXT: Is migrated to (with the option --use-experimental-features=virtual_mem):
// cuMemUnmap-NEXT:   sycl::ext::oneapi::experimental::unmap(ptr, size, dpct::get_current_device().get_context());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemCreate | FileCheck %s -check-prefix=cuMemCreate
// cuMemCreate: CUDA API:
// cuMemCreate-NEXT:   CUmemGenericAllocationHandle *handle;
// cuMemCreate-NEXT:   cuMemCreate(handle /*CUmemGenericAllocationHandle **/, size /*size_t*/,
// cuMemCreate-NEXT:               prop /*CUmemAllocationProp **/, flags /*unsigned long long*/);
// cuMemCreate-NEXT: Is migrated to (with the option --use-experimental-features=virtual_mem):
// cuMemCreate-NEXT:   dpct::experimental::physical_mem_ptr *handle;
// cuMemCreate-NEXT:   *handle = new sycl::ext::oneapi::experimental::physical_mem(dpct::get_device(prop->location.id), dpct::get_device(prop->location.id).get_context(), size);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAddressFree | FileCheck %s -check-prefix=cuMemAddressFree
// cuMemAddressFree: CUDA API:
// cuMemAddressFree-NEXT:   cuMemAddressFree(ptr /*CUdeviceptr*/, size /*size_t*/);
// cuMemAddressFree-NEXT: Is migrated to (with the option --use-experimental-features=virtual_mem):
// cuMemAddressFree-NEXT:   sycl::ext::oneapi::experimental::free_virtual_mem((uintptr_t)ptr, size, dpct::get_current_device().get_context());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemSetAccess | FileCheck %s -check-prefix=cuMemSetAccess
// cuMemSetAccess: CUDA API:
// cuMemSetAccess-NEXT:   cuMemSetAccess(ptr /*CUdeviceptr*/, size /*size_t*/,
// cuMemSetAccess-NEXT:                  desc /*CUmemAccessDesc **/, count /*size_t*/);
// cuMemSetAccess-NEXT: Is migrated to (with the option --use-experimental-features=virtual_mem):
// cuMemSetAccess-NEXT:   sycl::ext::oneapi::experimental::set_access_mode(ptr, size, desc->flags, dpct::get_device(desc->location.id).get_context());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemGetAllocationGranularity | FileCheck %s -check-prefix=cuMemGetAllocationGranularity
// cuMemGetAllocationGranularity: CUDA API:
// cuMemGetAllocationGranularity-NEXT:   cuMemGetAllocationGranularity(granularity /*size_t
// cuMemGetAllocationGranularity-NEXT:                                              **/
// cuMemGetAllocationGranularity-NEXT:                                 ,
// cuMemGetAllocationGranularity-NEXT:                                 prop /*CUmemAllocationProp **/,
// cuMemGetAllocationGranularity-NEXT:                                 option /*CUmemAllocationGranularity_flags*/);
// cuMemGetAllocationGranularity-NEXT: Is migrated to (with the option --use-experimental-features=virtual_mem):
// cuMemGetAllocationGranularity-NEXT:   *granularity = sycl::ext::oneapi::experimental::get_mem_granularity(dpct::get_device(prop->location.id), dpct::get_device(prop->location.id).get_context(), option);
