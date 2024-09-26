/// Error Handling

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuGetErrorString | FileCheck %s -check-prefix=CUGETERRORSTRING
// CUGETERRORSTRING: CUDA API:
// CUGETERRORSTRING-NEXT:   cuGetErrorString(r /*CUresult*/, ppc /*const char ***/);
// CUGETERRORSTRING-NEXT: Is migrated to:
// CUGETERRORSTRING-NEXT:   /*
// CUGETERRORSTRING-NEXT:   DPCT1009:0: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced by a placeholder string. You need to rewrite this code.
// CUGETERRORSTRING-NEXT:   */
// CUGETERRORSTRING-NEXT:   *ppc = "<Placeholder string>";

/// Initialization

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuInit | FileCheck %s -check-prefix=CUINIT
// CUINIT: CUDA API:
// CUINIT-NEXT:   cuInit(u /*unsigned int*/);
// CUINIT-NEXT: The API is Removed.
// CUINIT-EMPTY:

/// Version Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDriverGetVersion | FileCheck %s -check-prefix=CUDRIVERGETVERSION
// CUDRIVERGETVERSION: CUDA API:
// CUDRIVERGETVERSION-NEXT:   cuDriverGetVersion(pi /*int **/);
// CUDRIVERGETVERSION-NEXT: Is migrated to:
// CUDRIVERGETVERSION-NEXT:   *pi = std::stoi(dpct::get_current_device().get_info<sycl::info::device::version>());

/// Device Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGet | FileCheck %s -check-prefix=CUDEVICEGET
// CUDEVICEGET: CUDA API:
// CUDEVICEGET-NEXT:   cuDeviceGet(pd /*CUdevice **/, i /*int*/);
// CUDEVICEGET-NEXT: Is migrated to:
// CUDEVICEGET-NEXT:   *pd = i;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetAttribute | FileCheck %s -check-prefix=CUDEVICEGETATTRIBUTE
// CUDEVICEGETATTRIBUTE: CUDA API:
// CUDEVICEGETATTRIBUTE:   /* 1 */ cuDeviceGetAttribute(
// CUDEVICEGETATTRIBUTE:       pi /*int **/, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 2 */ cuDeviceGetAttribute(
// CUDEVICEGETATTRIBUTE:       pi /*int **/, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 3 */ cuDeviceGetAttribute(
// CUDEVICEGETATTRIBUTE:       pi /*int **/, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 4 */ cuDeviceGetAttribute(
// CUDEVICEGETATTRIBUTE:       pi /*int **/, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 5 */ cuDeviceGetAttribute(pi /*int **/,
// CUDEVICEGETATTRIBUTE:                                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
// CUDEVICEGETATTRIBUTE:                                d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 6 */ cuDeviceGetAttribute(
// CUDEVICEGETATTRIBUTE:       pi /*int **/, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 7 */ cuDeviceGetAttribute(pi /*int **/, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
// CUDEVICEGETATTRIBUTE:                                d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 8 */ cuDeviceGetAttribute(pi /*int **/,
// CUDEVICEGETATTRIBUTE:                                CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
// CUDEVICEGETATTRIBUTE:                                d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 9 */ cuDeviceGetAttribute(pi /*int **/, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
// CUDEVICEGETATTRIBUTE:                                d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 10 */ cuDeviceGetAttribute(
// CUDEVICEGETATTRIBUTE:       pi /*int **/, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 11 */ cuDeviceGetAttribute(
// CUDEVICEGETATTRIBUTE:       pi /*int **/, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 12 */ cuDeviceGetAttribute(pi /*int **/, CU_DEVICE_ATTRIBUTE_INTEGRATED,
// CUDEVICEGETATTRIBUTE:                                 d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 13 */ cuDeviceGetAttribute(
// CUDEVICEGETATTRIBUTE:       pi /*int **/, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 14 */ cuDeviceGetAttribute(pi /*int **/,
// CUDEVICEGETATTRIBUTE:                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
// CUDEVICEGETATTRIBUTE:                                 d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 15 */ cuDeviceGetAttribute(pi /*int **/,
// CUDEVICEGETATTRIBUTE:                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
// CUDEVICEGETATTRIBUTE:                                 d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE:   /* 16 */ cuDeviceGetAttribute(
// CUDEVICEGETATTRIBUTE:       pi /*int **/, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
// CUDEVICEGETATTRIBUTE:       d /*CUdevice*/);
// CUDEVICEGETATTRIBUTE-NEXT: Is migrated to:
// CUDEVICEGETATTRIBUTE-NEXT:  /* 1 */ *pi = dpct::get_device(d).get_max_work_group_size();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 2 */ *pi = dpct::get_device(d).get_device_info().get_max_work_item_sizes().get(0);
// CUDEVICEGETATTRIBUTE-NEXT:  /* 3 */ *pi = dpct::get_device(d).get_device_info().get_max_work_item_sizes().get(1);
// CUDEVICEGETATTRIBUTE-NEXT:  /* 4 */ *pi = dpct::get_device(d).get_device_info().get_max_work_item_sizes().get(2);
// CUDEVICEGETATTRIBUTE-NEXT:  /* 5 */ *pi = dpct::get_device(d).get_device_info().get_local_mem_size();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 6 */ *pi = dpct::get_device(d).get_global_mem_size();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 7 */ *pi = dpct::get_device(d).get_max_sub_group_size();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 8 */ *pi = dpct::get_device(d).get_max_register_size_per_work_group();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 9 */ *pi = dpct::get_device(d).get_max_clock_frequency();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 10 */ *pi = dpct::get_device(d).get_mem_base_addr_align_in_bytes();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 11 */ *pi = dpct::get_device(d).get_max_compute_units();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 12 */ *pi = dpct::get_device(d).get_integrated();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 13 */ *pi = dpct::get_device(d).has(sycl::aspect::usm_host_allocations);
// CUDEVICEGETATTRIBUTE-NEXT:  /* 14 */ *pi = dpct::get_device(d).get_major_version();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 15 */ *pi = dpct::get_device(d).get_minor_version();
// CUDEVICEGETATTRIBUTE-NEXT:  /* 16 */ *pi = dpct::get_device(d).is_native_atomic_supported();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetCount | FileCheck %s -check-prefix=CUDEVICEGETCOUNT
// CUDEVICEGETCOUNT: CUDA API:
// CUDEVICEGETCOUNT-NEXT:   cuDeviceGetCount(pi /*int **/);
// CUDEVICEGETCOUNT-NEXT: Is migrated to:
// CUDEVICEGETCOUNT-NEXT:   *pi = dpct::device_count();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetName | FileCheck %s -check-prefix=CUDEVICEGETNAME
// CUDEVICEGETNAME: CUDA API:
// CUDEVICEGETNAME-NEXT:   cuDeviceGetName(pc /*char **/, i /*int*/, d /*CUdevice*/);
// CUDEVICEGETNAME-NEXT: Is migrated to:
// CUDEVICEGETNAME-NEXT:   memcpy(pc, dpct::get_device(d).get_info<sycl::info::device::name>().c_str(), i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceTotalMem | FileCheck %s -check-prefix=CUDEVICETOTALMEM
// CUDEVICETOTALMEM: CUDA API:
// CUDEVICETOTALMEM-NEXT:   cuDeviceTotalMem(ps /*size_t **/, d /*CUdevice*/);
// CUDEVICETOTALMEM-NEXT: Is migrated to:
// CUDEVICETOTALMEM-NEXT:   *ps = dpct::get_device(d).get_device_info().get_global_mem_size();

/// Device Management [DEPRECATED]

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceComputeCapability | FileCheck %s -check-prefix=CUDEVICECOMPUTECAPABILITY
// CUDEVICECOMPUTECAPABILITY: CUDA API:
// CUDEVICECOMPUTECAPABILITY-NEXT:   cuDeviceComputeCapability(pi1 /*int **/, pi2 /*int **/, d /*CUdevice*/);
// CUDEVICECOMPUTECAPABILITY-NEXT: Is migrated to:
// CUDEVICECOMPUTECAPABILITY-NEXT:   *pi1 = dpct::get_major_version(dpct::get_device(d));
// CUDEVICECOMPUTECAPABILITY-NEXT:   *pi2 = dpct::get_minor_version(dpct::get_device(d));

/// Primary Context Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDevicePrimaryCtxRelease | FileCheck %s -check-prefix=CUDEVICEPRIMARYCTXRELEASE
// CUDEVICEPRIMARYCTXRELEASE: CUDA API:
// CUDEVICEPRIMARYCTXRELEASE-NEXT:   cuDevicePrimaryCtxRelease(d /*CUdevice*/);
// CUDEVICEPRIMARYCTXRELEASE-NEXT: The API is Removed.
// CUDEVICEPRIMARYCTXRELEASE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDevicePrimaryCtxRetain | FileCheck %s -check-prefix=CUDEVICEPRIMARYCTXRETAIN
// CUDEVICEPRIMARYCTXRETAIN: CUDA API:
// CUDEVICEPRIMARYCTXRETAIN-NEXT:   cuDevicePrimaryCtxRetain(pc /*CUcontext **/, d /*CUdevice*/);
// CUDEVICEPRIMARYCTXRETAIN-NEXT: Is migrated to:
// CUDEVICEPRIMARYCTXRETAIN-NEXT:   *pc = dpct::select_device(d);

/// Context Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxCreate | FileCheck %s -check-prefix=CUCTXCREATE
// CUCTXCREATE: CUDA API:
// CUCTXCREATE-NEXT:   cuCtxCreate(pc /*CUcontext **/, u /*unsigned*/, d /*CUdevice*/);
// CUCTXCREATE-NEXT: Is migrated to:
// CUCTXCREATE-NEXT:   *pc = dpct::push_device_for_curr_thread(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxDestroy | FileCheck %s -check-prefix=CUCTXDESTROY
// CUCTXDESTROY: CUDA API:
// CUCTXDESTROY-NEXT:   cuCtxDestroy(c /*CUcontext*/);
// CUCTXDESTROY-NEXT: The API is Removed.
// CUCTXDESTROY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxGetApiVersion | FileCheck %s -check-prefix=CUCTXGETAPIVERSION
// CUCTXGETAPIVERSION: CUDA API:
// CUCTXGETAPIVERSION-NEXT:   cuCtxGetApiVersion(c /*CUcontext*/, u /*unsigned int **/);
// CUCTXGETAPIVERSION-NEXT: Is migrated to:
// CUCTXGETAPIVERSION-NEXT:   *u = dpct::get_sycl_language_version();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxGetCurrent | FileCheck %s -check-prefix=CUCTXGETCURRENT
// CUCTXGETCURRENT: CUDA API:
// CUCTXGETCURRENT-NEXT:   cuCtxGetCurrent(pc /*CUcontext **/);
// CUCTXGETCURRENT-NEXT: Is migrated to:
// CUCTXGETCURRENT-NEXT:   *pc = dpct::get_current_device_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxGetDevice | FileCheck %s -check-prefix=CUCTXGETDEVICE
// CUCTXGETDEVICE: CUDA API:
// CUCTXGETDEVICE-NEXT:   cuCtxGetDevice(pd /*CUdevice **/);
// CUCTXGETDEVICE-NEXT: Is migrated to:
// CUCTXGETDEVICE-NEXT:   *pd = dpct::get_current_device_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxPushCurrent | FileCheck %s -check-prefix=CUCTXPUSHCURRENT
// CUCTXPUSHCURRENT: CUDA API:
// CUCTXPUSHCURRENT-NEXT:   cuCtxPushCurrent(c /*CUcontext*/);
// CUCTXPUSHCURRENT-NEXT: Is migrated to:
// CUCTXPUSHCURRENT-NEXT:   dpct::push_device_for_curr_thread(c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxPopCurrent | FileCheck %s -check-prefix=CUCTXPOPCURRENT
// CUCTXPOPCURRENT: CUDA API:
// CUCTXPOPCURRENT-NEXT:   cuCtxPopCurrent(&c /*CUcontext **/);
// CUCTXPOPCURRENT-NEXT: Is migrated to:
// CUCTXPOPCURRENT-NEXT:   c = dpct::pop_device_for_curr_thread();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxGetLimit | FileCheck %s -check-prefix=CUCTXGETLIMIT
// CUCTXGETLIMIT: CUDA API:
// CUCTXGETLIMIT-NEXT:   cuCtxGetLimit(ps /*size_t **/, CU_LIMIT_PRINTF_FIFO_SIZE);
// CUCTXGETLIMIT-NEXT: Is migrated to:
// CUCTXGETLIMIT-NEXT:   *ps = INT_MAX;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxSetCacheConfig | FileCheck %s -check-prefix=CUCTXSETCACHECONFIG
// CUCTXSETCACHECONFIG: CUDA API:
// CUCTXSETCACHECONFIG-NEXT:   cuCtxSetCacheConfig(f /*CUfunc_cache*/);
// CUCTXSETCACHECONFIG-NEXT: The API is Removed.
// CUCTXSETCACHECONFIG-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxSetCurrent | FileCheck %s -check-prefix=CUCTXSETCURRENT
// CUCTXSETCURRENT: CUDA API:
// CUCTXSETCURRENT-NEXT:   cuCtxSetCurrent(c /*CUcontext*/);
// CUCTXSETCURRENT-NEXT: Is migrated to:
// CUCTXSETCURRENT-NEXT:   dpct::select_device(c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxSetLimit | FileCheck %s -check-prefix=CUCTXSETLIMIT
// CUCTXSETLIMIT: CUDA API:
// CUCTXSETLIMIT-NEXT:   cuCtxSetLimit(l /*CUlimit*/, s /*size_t*/);
// CUCTXSETLIMIT-NEXT: The API is Removed.
// CUCTXSETLIMIT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxSynchronize | FileCheck %s -check-prefix=CUCTXSYNCHRONIZE
// CUCTXSYNCHRONIZE: CUDA API:
// CUCTXSYNCHRONIZE-NEXT:   cuCtxSynchronize();
// CUCTXSYNCHRONIZE-NEXT: Is migrated to:
// CUCTXSYNCHRONIZE-NEXT:   dpct::get_current_device().queues_wait_and_throw();

/// Module Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleGetFunction | FileCheck %s -check-prefix=CUMODULEGETFUNCTION
// CUMODULEGETFUNCTION: CUDA API:
// CUMODULEGETFUNCTION-NEXT:   cuModuleGetFunction(pf /*CUfunction **/, m /*CUmodule*/, pc /*const char **/);
// CUMODULEGETFUNCTION-NEXT: Is migrated to:
// CUMODULEGETFUNCTION-NEXT:   *pf = dpct::get_kernel_function(m, pc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleLoad | FileCheck %s -check-prefix=CUMODULELOAD
// CUMODULELOAD: CUDA API:
// CUMODULELOAD-NEXT:   cuModuleLoad(pm /*CUmodule **/, pc /*const char **/);
// CUMODULELOAD-NEXT: Is migrated to:
// CUMODULELOAD-NEXT:   *pm = dpct::load_kernel_library(pc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleLoadData | FileCheck %s -check-prefix=CUMODULELOADDATA
// CUMODULELOADDATA: CUDA API:
// CUMODULELOADDATA-NEXT:   cuModuleLoadData(pm /*CUmodule **/, pData /*const void **/);
// CUMODULELOADDATA-NEXT: Is migrated to:
// CUMODULELOADDATA-NEXT:    *pm = dpct::load_kernel_library_mem(pData);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleLoadDataEx | FileCheck %s -check-prefix=CUMODULELOADDATAEX
// CUMODULELOADDATAEX: CUDA API:
// CUMODULELOADDATAEX-NEXT:   cuModuleLoadDataEx(pm /*CUmodule **/, pData /*const void **/,
// CUMODULELOADDATAEX-NEXT:                      u /*unsigned int*/, pOpt /*CUjit_option **/,
// CUMODULELOADDATAEX-NEXT:                      pOptVal /*void ***/);
// CUMODULELOADDATAEX-NEXT: Is migrated to:
// CUMODULELOADDATAEX-NEXT:   *pm = dpct::load_kernel_library_mem(pData);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleUnload | FileCheck %s -check-prefix=CUMODULEUNLOAD
// CUMODULEUNLOAD: CUDA API:
// CUMODULEUNLOAD-NEXT:   cuModuleUnload(m /*CUmodule*/);
// CUMODULEUNLOAD-NEXT: Is migrated to:
// CUMODULEUNLOAD-NEXT:   dpct::unload_kernel_library(m);

/// Module Management [DEPRECATED]

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleGetTexRef | FileCheck %s -check-prefix=CUMODULEGETTEXREF
// CUMODULEGETTEXREF: CUDA API:
// CUMODULEGETTEXREF-NEXT:   cuModuleGetTexRef(pt /*CUtexref **/, m /*CUmodule*/, pc /*const char **/);
// CUMODULEGETTEXREF-NEXT: Is migrated to:
// CUMODULEGETTEXREF-NEXT:   *pt = dpct::get_image_wrapper(m, pc);

/// Unified Addressing

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAdvise | FileCheck %s -check-prefix=CUMEMADVISE
// CUMEMADVISE: CUDA API:
// CUMEMADVISE-NEXT:   cuMemAdvise(pd /*CUdeviceptr*/, s /*size_t*/, m /*CUmem_advise*/,
// CUMEMADVISE-NEXT:               d /*CUdevice*/);
// CUMEMADVISE-NEXT: Is migrated to:
// CUMEMADVISE-NEXT:   dpct::get_device(d).in_order_queue().mem_advise(pd, s, m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemPrefetchAsync | FileCheck %s -check-prefix=CUMEMPREFETCHASYNC
// CUMEMPREFETCHASYNC: CUDA API:
// CUMEMPREFETCHASYNC-NEXT:   CUstream cs;
// CUMEMPREFETCHASYNC-NEXT:   cuMemPrefetchAsync(pd /*CUdeviceptr*/, s /*size_t*/, d /*CUdevice*/,
// CUMEMPREFETCHASYNC-NEXT:                      cs /*CUstream*/);
// CUMEMPREFETCHASYNC-NEXT: Is migrated to:
// CUMEMPREFETCHASYNC-NEXT:   dpct::queue_ptr cs;
// CUMEMPREFETCHASYNC-NEXT:   cs->prefetch(pd, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuPointerGetAttributes | FileCheck %s -check-prefix=CUPOINTERGETATTRIBUTES
// CUPOINTERGETATTRIBUTES: CUDA API:
// CUPOINTERGETATTRIBUTES-NEXT:   cuPointerGetAttributes(numAttr /*unsigned int*/,
// CUPOINTERGETATTRIBUTES-NEXT:                            attr /*CUpointer_attribute **/, data /*void ***/,
// CUPOINTERGETATTRIBUTES-NEXT:                            ptr /*CUdeviceptr*/);
// CUPOINTERGETATTRIBUTES-NEXT: Is migrated to:
// CUPOINTERGETATTRIBUTES-NEXT:   dpct::pointer_attributes::get(numAttr, attr, data, ptr);

/// Stream Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamAddCallback | FileCheck %s -check-prefix=CUSTREAMADDCALLBACK
// CUSTREAMADDCALLBACK: CUDA API:
// CUSTREAMADDCALLBACK-NEXT:   CUstream s;
// CUSTREAMADDCALLBACK-NEXT:   cuStreamAddCallback(s /*CUstream*/, sc /*CUstreamCallback*/, pData /*void **/,
// CUSTREAMADDCALLBACK-NEXT:                       u /*unsigned int*/);
// CUSTREAMADDCALLBACK-NEXT: Is migrated to:
// CUSTREAMADDCALLBACK-NEXT:   dpct::queue_ptr s;
// CUSTREAMADDCALLBACK-NEXT:   std::async([&]() { s->wait(); sc(s, 0, pData); });

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamAttachMemAsync | FileCheck %s -check-prefix=CUSTREAMATTACHMEMASYNC
// CUSTREAMATTACHMEMASYNC: CUDA API:
// CUSTREAMATTACHMEMASYNC-NEXT:   cuStreamAttachMemAsync(cs /*CUstream*/, d /*CUdeviceptr*/, s /*size_t*/,
// CUSTREAMATTACHMEMASYNC-NEXT:                          u /*unsigned int*/);
// CUSTREAMATTACHMEMASYNC-NEXT: The API is Removed.
// CUSTREAMATTACHMEMASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamCreate | FileCheck %s -check-prefix=CUSTREAMCREATE
// CUSTREAMCREATE: CUDA API:
// CUSTREAMCREATE-NEXT:   cuStreamCreate(ps /*CUstream **/, u /*unsigned int*/);
// CUSTREAMCREATE-NEXT: Is migrated to:
// CUSTREAMCREATE-NEXT:   *(ps) = dpct::get_current_device().create_queue();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamDestroy | FileCheck %s -check-prefix=CUSTREAMDESTROY
// CUSTREAMDESTROY: CUDA API:
// CUSTREAMDESTROY-NEXT:   cuStreamDestroy(s /*CUstream*/);
// CUSTREAMDESTROY-NEXT: Is migrated to:
// CUSTREAMDESTROY-NEXT:   dpct::get_current_device().destroy_queue(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamQuery | FileCheck %s -check-prefix=CUSTREAMQUERY
// CUSTREAMQUERY: CUDA API:
// CUSTREAMQUERY-NEXT:   CUstream s;
// CUSTREAMQUERY-NEXT:   cuStreamQuery(s);
// CUSTREAMQUERY-NEXT: Is migrated to:
// CUSTREAMQUERY-NEXT:   dpct::queue_ptr s;
// CUSTREAMQUERY-NEXT:   s->ext_oneapi_empty();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamSynchronize | FileCheck %s -check-prefix=CUSTREAMSYNCHRONIZE
// CUSTREAMSYNCHRONIZE: CUDA API:
// CUSTREAMSYNCHRONIZE-NEXT:   CUstream s;
// CUSTREAMSYNCHRONIZE-NEXT:   cuStreamSynchronize(s /*CUstream*/);
// CUSTREAMSYNCHRONIZE-NEXT: Is migrated to:
// CUSTREAMSYNCHRONIZE-NEXT:   dpct::queue_ptr s;
// CUSTREAMSYNCHRONIZE-NEXT:   s->wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamWaitEvent | FileCheck %s -check-prefix=CUSTREAMWAITEVENT
// CUSTREAMWAITEVENT: CUDA API:
// CUSTREAMWAITEVENT-NEXT:   CUstream s;
// CUSTREAMWAITEVENT-NEXT:   cuStreamWaitEvent(s /*CUstream*/, e /*CUevent*/, u /*unsigned int*/);
// CUSTREAMWAITEVENT-NEXT: Is migrated to:
// CUSTREAMWAITEVENT-NEXT:   dpct::queue_ptr s;
// CUSTREAMWAITEVENT-NEXT:   s->ext_oneapi_submit_barrier({*e});

/// Event Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventCreate | FileCheck %s -check-prefix=CUEVENTCREATE
// CUEVENTCREATE: CUDA API:
// CUEVENTCREATE-NEXT:   cuEventCreate(pe /*CUevent **/, u /*unsigned int*/);
// CUEVENTCREATE-NEXT: Is migrated to:
// CUEVENTCREATE-NEXT:   *pe = new sycl::event();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventDestroy | FileCheck %s -check-prefix=CUEVENTDESTROY
// CUEVENTDESTROY: CUDA API:
// CUEVENTDESTROY-NEXT:   cuEventDestroy(e /*CUevent*/);
// CUEVENTDESTROY-NEXT: Is migrated to:
// CUEVENTDESTROY-NEXT:   dpct::destroy_event(e);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventElapsedTime | FileCheck %s -check-prefix=CUEVENTELAPSEDTIME
// CUEVENTELAPSEDTIME: CUDA API:
// CUEVENTELAPSEDTIME-NEXT:   cuEventElapsedTime(pf /*float **/, e1 /*CUevent*/, e2 /*CUevent*/);
// CUEVENTELAPSEDTIME-NEXT: Is migrated to:
// CUEVENTELAPSEDTIME-NEXT:   *(pf) = std::chrono::duration<float, std::milli>(e2_ct1 - e1_ct1).count();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventQuery | FileCheck %s -check-prefix=CUEVENTQUERY
// CUEVENTQUERY: CUDA API:
// CUEVENTQUERY-NEXT:   CUevent e;
// CUEVENTQUERY-NEXT:   cuEventQuery(e /*CUevent*/);
// CUEVENTQUERY-NEXT: Is migrated to:
// CUEVENTQUERY-NEXT:   dpct::event_ptr e;
// CUEVENTQUERY-NEXT:   dpct::sycl_event_query(e /*CUevent*/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventRecord | FileCheck %s -check-prefix=CUEVENTRECORD
// CUEVENTRECORD: CUDA API:
// CUEVENTRECORD-NEXT:   CUstream s;
// CUEVENTRECORD-NEXT:   cuEventRecord(e /*CUevent*/, s /*CUstream*/);
// CUEVENTRECORD-NEXT: Is migrated to:
// CUEVENTRECORD-NEXT:   dpct::queue_ptr s;
// CUEVENTRECORD-NEXT:   *e = s->ext_oneapi_submit_barrier();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventSynchronize | FileCheck %s -check-prefix=CUEVENTSYNCHRONIZE
// CUEVENTSYNCHRONIZE: CUDA API:
// CUEVENTSYNCHRONIZE-NEXT:   CUevent e;
// CUEVENTSYNCHRONIZE-NEXT:   cuEventSynchronize(e /*CUevent*/);
// CUEVENTSYNCHRONIZE-NEXT: Is migrated to:
// CUEVENTSYNCHRONIZE-NEXT:   dpct::event_ptr e;
// CUEVENTSYNCHRONIZE-NEXT:   e->wait_and_throw();

/// Execution Control

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuFuncGetAttribute | FileCheck %s -check-prefix=CUFUNCGETATTRIBUTE
// CUFUNCGETATTRIBUTE: CUDA API:
// CUFUNCGETATTRIBUTE-NEXT:   cuFuncGetAttribute(pi /*int **/, fa /*CUfunction_attribute*/,
// CUFUNCGETATTRIBUTE-NEXT:                      f /*CUfunction*/);
// CUFUNCGETATTRIBUTE-NEXT: Is migrated to:
// CUFUNCGETATTRIBUTE-NEXT:   *pi = dpct::get_kernel_function_info(f).fa;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuFuncSetCacheConfig | FileCheck %s -check-prefix=CUFUNCSETCACHECONFIG
// CUFUNCSETCACHECONFIG: CUDA API:
// CUFUNCSETCACHECONFIG-NEXT:   cuFuncSetCacheConfig(f /*CUfunction*/, fc /*CUfunc_cache*/);
// CUFUNCSETCACHECONFIG-NEXT: The API is Removed.
// CUFUNCSETCACHECONFIG-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuLaunchKernel | FileCheck %s -check-prefix=CULAUNCHKERNEL
// CULAUNCHKERNEL: CUDA API:
// CULAUNCHKERNEL-NEXT:   cuLaunchKernel(f /*CUfunction*/, u1 /*unsigned int*/, u2 /*unsigned int*/,
// CULAUNCHKERNEL-NEXT:                  u3 /*unsigned int*/, u4 /*unsigned int*/, u5 /*unsigned int*/,
// CULAUNCHKERNEL-NEXT:                  u6 /*unsigned int*/, u7 /*unsigned int*/, s /*CUstream*/,
// CULAUNCHKERNEL-NEXT:                  pParam /*void ***/, pOpt /*void ***/);
// CULAUNCHKERNEL-NEXT: Is migrated to:
// CULAUNCHKERNEL-NEXT:   dpct::invoke_kernel_function(f, *s, sycl::range<3>(u3, u2, u1), sycl::range<3>(u6, u5, u4), u7, pParam, pOpt);

/// Occupancy

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuOccupancyMaxActiveBlocksPerMultiprocessor | FileCheck %s -check-prefix=CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR
// CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR: CUDA API:
// CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR-NEXT:   cuOccupancyMaxActiveBlocksPerMultiprocessor(pi /*int **/, f /*CUfunction*/,
// CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR-NEXT:                                               i /*int*/, s /*size_t*/);
// CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR-NEXT: Is migrated to (with the option --use-experimental-features=occupancy-calculation):
// CUOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR-NEXT:   dpct::experimental::calculate_max_active_wg_per_xecore(pi, i, s + dpct_placeholder /* total share local memory size */);

/// Texture Reference Management [DEPRECATED]

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefGetAddressMode | FileCheck %s -check-prefix=CUTEXREFGETADDRESSMODE
// CUTEXREFGETADDRESSMODE: CUDA API:
// CUTEXREFGETADDRESSMODE-NEXT:   CUtexref t;
// CUTEXREFGETADDRESSMODE-NEXT:   cuTexRefGetAddressMode(pa /*CUaddress_mode **/, t /*CUtexref*/, i /*int*/);
// CUTEXREFGETADDRESSMODE-NEXT: Is migrated to:
// CUTEXREFGETADDRESSMODE-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFGETADDRESSMODE-NEXT:   *pa = t->get_addressing_mode();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefGetFilterMode | FileCheck %s -check-prefix=CUTEXREFGETFILTERMODE
// CUTEXREFGETFILTERMODE: CUDA API:
// CUTEXREFGETFILTERMODE-NEXT:   CUtexref t;
// CUTEXREFGETFILTERMODE-NEXT:   cuTexRefGetFilterMode(pf /*CUfilter_mode **/, t /*CUtexref*/);
// CUTEXREFGETFILTERMODE-NEXT: Is migrated to:
// CUTEXREFGETFILTERMODE-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFGETFILTERMODE-NEXT:   *pf = t->get_filtering_mode();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefGetFlags | FileCheck %s -check-prefix=CUTEXREFGETFLAGS
// CUTEXREFGETFLAGS: CUDA API:
// CUTEXREFGETFLAGS-NEXT:   CUtexref t;
// CUTEXREFGETFLAGS-NEXT:   cuTexRefGetFlags(pu /*unsigned int **/, t /*CUtexref*/);
// CUTEXREFGETFLAGS-NEXT: Is migrated to:
// CUTEXREFGETFLAGS-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFGETFLAGS-NEXT:   *pu = t->is_coordinate_normalized() << 1;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefSetAddress | FileCheck %s -check-prefix=CUTEXREFSETADDRESS
// CUTEXREFSETADDRESS: CUDA API:
// CUTEXREFSETADDRESS-NEXT:   CUtexref t;
// CUTEXREFSETADDRESS-NEXT:   cuTexRefSetAddress(ps /*size_t **/, t /*CUtexref*/, d /*CUdeviceptr*/,
// CUTEXREFSETADDRESS-NEXT:                      s /*size_t*/);
// CUTEXREFSETADDRESS-NEXT: Is migrated to:
// CUTEXREFSETADDRESS-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFSETADDRESS-NEXT:   t->attach(d, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefSetAddress2D | FileCheck %s -check-prefix=CUTEXREFSETADDRESS2D
// CUTEXREFSETADDRESS2D: CUDA API:
// CUTEXREFSETADDRESS2D-NEXT:   CUtexref t;
// CUTEXREFSETADDRESS2D-NEXT:   cuTexRefSetAddress2D(t /*CUtexref*/, pa /*const CUDA_ARRAY_DESCRIPTOR **/,
// CUTEXREFSETADDRESS2D-NEXT:                        d /*CUdeviceptr*/, s /*size_t*/);
// CUTEXREFSETADDRESS2D-NEXT: Is migrated to:
// CUTEXREFSETADDRESS2D-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFSETADDRESS2D-NEXT:   t->attach(pa, d, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefSetAddressMode | FileCheck %s -check-prefix=CUTEXREFSETADDRESSMODE
// CUTEXREFSETADDRESSMODE: CUDA API:
// CUTEXREFSETADDRESSMODE-NEXT:   CUtexref t;
// CUTEXREFSETADDRESSMODE-NEXT:   cuTexRefSetAddressMode(t /*CUtexref*/, i /*int **/, a /*CUaddress_mode*/);
// CUTEXREFSETADDRESSMODE-NEXT: Is migrated to:
// CUTEXREFSETADDRESSMODE-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFSETADDRESSMODE-NEXT:   t->set(a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefSetArray | FileCheck %s -check-prefix=CUTEXREFSETARRAY
// CUTEXREFSETARRAY: CUDA API:
// CUTEXREFSETARRAY-NEXT:   CUtexref t;
// CUTEXREFSETARRAY-NEXT:   cuTexRefSetArray(t /*CUtexref*/, a /*CUarray*/, u /*unsigned int*/);
// CUTEXREFSETARRAY-NEXT: Is migrated to:
// CUTEXREFSETARRAY-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFSETARRAY-NEXT:   t->attach(dpct::image_data(a));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefSetFilterMode | FileCheck %s -check-prefix=CUTEXREFSETFILTERMODE
// CUTEXREFSETFILTERMODE: CUDA API:
// CUTEXREFSETFILTERMODE-NEXT:   CUtexref t;
// CUTEXREFSETFILTERMODE-NEXT:   cuTexRefSetFilterMode(t /*CUtexref*/, f /*CUfilter_mode*/);
// CUTEXREFSETFILTERMODE-NEXT: Is migrated to:
// CUTEXREFSETFILTERMODE-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFSETFILTERMODE-NEXT:   t->set(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefSetFlags | FileCheck %s -check-prefix=CUTEXREFSETFLAGS
// CUTEXREFSETFLAGS: CUDA API:
// CUTEXREFSETFLAGS-NEXT:   CUtexref t;
// CUTEXREFSETFLAGS-NEXT:   cuTexRefSetFlags(t /*CUtexref*/, u /*unsigned int*/);
// CUTEXREFSETFLAGS-NEXT: Is migrated to:
// CUTEXREFSETFLAGS-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFSETFLAGS-NEXT:   t->set_coordinate_normalization_mode(u & 0x02);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexRefSetFormat | FileCheck %s -check-prefix=CUTEXREFSETFORMAT
// CUTEXREFSETFORMAT: CUDA API:
// CUTEXREFSETFORMAT-NEXT:   CUtexref t;
// CUTEXREFSETFORMAT-NEXT:   cuTexRefSetFormat(t /*CUtexref*/, a /*CUarray_format*/, i /*int*/);
// CUTEXREFSETFORMAT-NEXT: Is migrated to:
// CUTEXREFSETFORMAT-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFSETFORMAT-NEXT:   t->set_channel_type(a);
// CUTEXREFSETFORMAT-NEXT:   t->set_channel_num(i);

/// Texture Object Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexObjectCreate | FileCheck %s -check-prefix=CUTEXOBJECTCREATE
// CUTEXOBJECTCREATE: CUDA API:
// CUTEXOBJECTCREATE-NEXT:   cuTexObjectCreate(pt /*CUtexObject **/, pr /*const CUDA_RESOURCE_DESC **/,
// CUTEXOBJECTCREATE-NEXT:                     ptd /*const CUDA_TEXTURE_DESC **/,
// CUTEXOBJECTCREATE-NEXT:                     prv /*const CUDA_RESOURCE_VIEW_DESC **/);
// CUTEXOBJECTCREATE-NEXT: Is migrated to:
// CUTEXOBJECTCREATE-NEXT:   *pt = dpct::create_image_wrapper(*pr, *ptd);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexObjectDestroy | FileCheck %s -check-prefix=CUTEXOBJECTDESTROY
// CUTEXOBJECTDESTROY: CUDA API:
// CUTEXOBJECTDESTROY-NEXT:   cuTexObjectDestroy(t /*CUtexObject*/);
// CUTEXOBJECTDESTROY-NEXT: Is migrated to:
// CUTEXOBJECTDESTROY-NEXT:   delete t;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexObjectGetResourceDesc | FileCheck %s -check-prefix=CUTEXOBJECTGETRESOURCEDESC
// CUTEXOBJECTGETRESOURCEDESC: CUDA API:
// CUTEXOBJECTGETRESOURCEDESC-NEXT:   CUtexObject t;
// CUTEXOBJECTGETRESOURCEDESC-NEXT:   cuTexObjectGetResourceDesc(pr /*CUDA_RESOURCE_DESC **/, t /*CUtexObject*/);
// CUTEXOBJECTGETRESOURCEDESC-NEXT: Is migrated to:
// CUTEXOBJECTGETRESOURCEDESC-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXOBJECTGETRESOURCEDESC-NEXT:   *pr = t->get_data();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuTexObjectGetTextureDesc | FileCheck %s -check-prefix=CUTEXOBJECTGETTEXTUREDESC
// CUTEXOBJECTGETTEXTUREDESC: CUDA API:
// CUTEXOBJECTGETTEXTUREDESC-NEXT:   CUtexObject t;
// CUTEXOBJECTGETTEXTUREDESC-NEXT:   cuTexObjectGetTextureDesc(pt /*CUDA_TEXTURE_DESC **/, t /*CUtexObject*/);
// CUTEXOBJECTGETTEXTUREDESC-NEXT: Is migrated to:
// CUTEXOBJECTGETTEXTUREDESC-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXOBJECTGETTEXTUREDESC-NEXT:   *pt = t->get_sampling_info();

/// Peer Context Memory Access

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxEnablePeerAccess | FileCheck %s -check-prefix=CUCTXENABLEPEERACCESS
// CUCTXENABLEPEERACCESS: CUDA API:
// CUCTXENABLEPEERACCESS-NEXT:   cuCtxEnablePeerAccess(c /*CUcontext*/, u /*unsigned*/);
// CUCTXENABLEPEERACCESS-NEXT: Is migrated to:
// CUCTXENABLEPEERACCESS-NEXT:   dpct::get_current_device().ext_oneapi_enable_peer_access(dpct::get_device(c));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceCanAccessPeer | FileCheck %s -check-prefix=CUDEVICECANACCESSPEER
// CUDEVICECANACCESSPEER: CUDA API:
// CUDEVICECANACCESSPEER-NEXT:   cuDeviceCanAccessPeer(pi /*int **/, d1 /*CUdevice*/, d2 /*CUdevice*/);
// CUDEVICECANACCESSPEER-NEXT: Is migrated to:
// CUDEVICECANACCESSPEER-NEXT:   *pi = dpct::get_device(d1).ext_oneapi_can_access_peer(dpct::get_device(d2));
