// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

/// Initialization

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuInit | FileCheck %s -check-prefix=CUINIT
// CUINIT: CUDA API:
// CUINIT-NEXT:   cuInit(u /*unsigned int*/);
// CUINIT-NEXT: The API is Removed.

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetCount | FileCheck %s -check-prefix=CUDEVICEGETCOUNT
// CUDEVICEGETCOUNT: CUDA API:
// CUDEVICEGETCOUNT-NEXT:   cuDeviceGetCount(pi /*int **/);
// CUDEVICEGETCOUNT-NEXT: Is migrated to:
// CUDEVICEGETCOUNT-NEXT:   *pi = dpct::dev_mgr::instance().device_count();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetName | FileCheck %s -check-prefix=CUDEVICEGETNAME
// CUDEVICEGETNAME: CUDA API:
// CUDEVICEGETNAME-NEXT:   cuDeviceGetName(pc /*char **/, i /*int*/, d /*CUdevice*/);
// CUDEVICEGETNAME-NEXT: Is migrated to:
// CUDEVICEGETNAME-NEXT:   memcpy(pc, dpct::dev_mgr::instance().get_device(d).get_info<sycl::info::device::name>().c_str(), i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceTotalMem | FileCheck %s -check-prefix=CUDEVICETOTALMEM
// CUDEVICETOTALMEM: CUDA API:
// CUDEVICETOTALMEM-NEXT:   cuDeviceTotalMem(ps /*size_t **/, d /*CUdevice*/);
// CUDEVICETOTALMEM-NEXT: Is migrated to:
// CUDEVICETOTALMEM-NEXT:   *ps = dpct::dev_mgr::instance().get_device(d).get_device_info().get_global_mem_size();

/// Device Management [DEPRECATED]

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceComputeCapability | FileCheck %s -check-prefix=CUDEVICECOMPUTECAPABILITY
// CUDEVICECOMPUTECAPABILITY: CUDA API:
// CUDEVICECOMPUTECAPABILITY-NEXT:   cuDeviceComputeCapability(pi1 /*int **/, pi2 /*int **/, d /*CUdevice*/);
// CUDEVICECOMPUTECAPABILITY-NEXT: Is migrated to:
// CUDEVICECOMPUTECAPABILITY-NEXT:   *pi1 = dpct::get_major_version(dpct::dev_mgr::instance().get_device(d));
// CUDEVICECOMPUTECAPABILITY-NEXT:   *pi2 = dpct::get_minor_version(dpct::dev_mgr::instance().get_device(d));

/// Primary Context Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDevicePrimaryCtxRelease | FileCheck %s -check-prefix=CUDEVICEPRIMARYCTXRELEASE
// CUDEVICEPRIMARYCTXRELEASE: CUDA API:
// CUDEVICEPRIMARYCTXRELEASE-NEXT:   cuDevicePrimaryCtxRelease(d /*CUdevice*/);
// CUDEVICEPRIMARYCTXRELEASE-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDevicePrimaryCtxRetain | FileCheck %s -check-prefix=CUDEVICEPRIMARYCTXRETAIN
// CUDEVICEPRIMARYCTXRETAIN: CUDA API:
// CUDEVICEPRIMARYCTXRETAIN-NEXT:   cuDevicePrimaryCtxRetain(pc /*CUcontext **/, d /*CUdevice*/);
// CUDEVICEPRIMARYCTXRETAIN-NEXT: Is migrated to:
// CUDEVICEPRIMARYCTXRETAIN-NEXT:   *pc = dpct::select_device(d);

/// Context Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxDestroy | FileCheck %s -check-prefix=CUCTXDESTROY
// CUCTXDESTROY: CUDA API:
// CUCTXDESTROY-NEXT:   cuCtxDestroy(c /*CUcontext*/);
// CUCTXDESTROY-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxGetApiVersion | FileCheck %s -check-prefix=CUCTXGETAPIVERSION
// CUCTXGETAPIVERSION: CUDA API:
// CUCTXGETAPIVERSION-NEXT:   cuCtxGetApiVersion(c /*CUcontext*/, u /*unsigned int **/);
// CUCTXGETAPIVERSION-NEXT: Is migrated to:
// CUCTXGETAPIVERSION-NEXT:   *u = dpct::get_sycl_language_version();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxGetCurrent | FileCheck %s -check-prefix=CUCTXGETCURRENT
// CUCTXGETCURRENT: CUDA API:
// CUCTXGETCURRENT-NEXT:   cuCtxGetCurrent(pc /*CUcontext **/);
// CUCTXGETCURRENT-NEXT: Is migrated to:
// CUCTXGETCURRENT-NEXT:   *pc = dpct::dev_mgr::instance().current_device_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxGetDevice | FileCheck %s -check-prefix=CUCTXGETDEVICE
// CUCTXGETDEVICE: CUDA API:
// CUCTXGETDEVICE-NEXT:   cuCtxGetDevice(pd /*CUdevice **/);
// CUCTXGETDEVICE-NEXT: Is migrated to:
// CUCTXGETDEVICE-NEXT:   *pd = dpct::get_current_device_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxSetCacheConfig | FileCheck %s -check-prefix=CUCTXSETCACHECONFIG
// CUCTXSETCACHECONFIG: CUDA API:
// CUCTXSETCACHECONFIG-NEXT:   cuCtxSetCacheConfig(f /*CUfunc_cache*/);
// CUCTXSETCACHECONFIG-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxSetCurrent | FileCheck %s -check-prefix=CUCTXSETCURRENT
// CUCTXSETCURRENT: CUDA API:
// CUCTXSETCURRENT-NEXT:   cuCtxSetCurrent(c /*CUcontext*/);
// CUCTXSETCURRENT-NEXT: Is migrated to:
// CUCTXSETCURRENT-NEXT:   dpct::select_device(c);

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

/// Memory Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuArrayDestroy | FileCheck %s -check-prefix=CUARRAYDESTROY
// CUARRAYDESTROY: CUDA API:
// CUARRAYDESTROY-NEXT:   cuArrayDestroy(a /*CUarray*/);
// CUARRAYDESTROY-NEXT: Is migrated to:
// CUARRAYDESTROY-NEXT:   delete a;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAlloc | FileCheck %s -check-prefix=CUMEMALLOC
// CUMEMALLOC: CUDA API:
// CUMEMALLOC-NEXT:   cuMemAlloc(pd /*CUdeviceptr **/, s /*size_t*/);
// CUMEMALLOC-NEXT: Is migrated to:
// CUMEMALLOC-NEXT:   *pd = (dpct::device_ptr)sycl::malloc_device(s, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAllocHost | FileCheck %s -check-prefix=CUMEMALLOCHOST
// CUMEMALLOCHOST: CUDA API:
// CUMEMALLOCHOST-NEXT:   cuMemAllocHost(pHost /*void ***/, s /*size_t*/);
// CUMEMALLOCHOST-NEXT: Is migrated to:
// CUMEMALLOCHOST-NEXT:   *pHost = (void *)sycl::malloc_host(s, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAllocManaged | FileCheck %s -check-prefix=CUMEMALLOCMANAGED
// CUMEMALLOCMANAGED: CUDA API:
// CUMEMALLOCMANAGED-NEXT:   cuMemAllocManaged(pd /*CUdeviceptr **/, s /*size_t*/, u /*unsigned int*/);
// CUMEMALLOCMANAGED-NEXT: Is migrated to:
// CUMEMALLOCMANAGED-NEXT:   *pd = (dpct::device_ptr)sycl::malloc_shared(s, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAllocPitch | FileCheck %s -check-prefix=CUMEMALLOCPITCH
// CUMEMALLOCPITCH: CUDA API:
// CUMEMALLOCPITCH-NEXT:   cuMemAllocPitch(pd /*CUdeviceptr **/, ps /*size_t **/, s1 /*size_t*/,
// CUMEMALLOCPITCH-NEXT:                   s2 /*size_t*/, u /*unsigned int*/);
// CUMEMALLOCPITCH-NEXT: Is migrated to:
// CUMEMALLOCPITCH-NEXT:   *pd = (dpct::device_ptr)dpct::dpct_malloc(*ps, s1, s2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemFree | FileCheck %s -check-prefix=CUMEMFREE
// CUMEMFREE: CUDA API:
// CUMEMFREE-NEXT:   cuMemFree(d /*CUdeviceptr*/);
// CUMEMFREE-NEXT: Is migrated to:
// CUMEMFREE-NEXT:   sycl::free(d, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemFreeHost | FileCheck %s -check-prefix=CUMEMFREEHOST
// CUMEMFREEHOST: CUDA API:
// CUMEMFREEHOST-NEXT:   cuMemFreeHost(pHost /*void **/);
// CUMEMFREEHOST-NEXT: Is migrated to:
// CUMEMFREEHOST-NEXT:   sycl::free(pHost, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemGetInfo | FileCheck %s -check-prefix=CUMEMGETINFO
// CUMEMGETINFO: CUDA API:
// CUMEMGETINFO-NEXT:   cuMemGetInfo(ps1 /*size_t **/, ps2 /*size_t **/);
// CUMEMGETINFO-NEXT: Is migrated to:
// CUMEMGETINFO-NEXT:   dpct::get_current_device().get_memory_info(*ps1, *ps2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostAlloc | FileCheck %s -check-prefix=CUMEMHOSTALLOC
// CUMEMHOSTALLOC: CUDA API:
// CUMEMHOSTALLOC-NEXT:   cuMemHostAlloc(pHost /*void ***/, s /*size_t*/, u /*unsigned int*/);
// CUMEMHOSTALLOC-NEXT: Is migrated to:
// CUMEMHOSTALLOC-NEXT:   *pHost = (void *)sycl::malloc_host(s, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostGetDevicePointer | FileCheck %s -check-prefix=CUMEMHOSTGETDEVICEPOINTER
// CUMEMHOSTGETDEVICEPOINTER: CUDA API:
// CUMEMHOSTGETDEVICEPOINTER-NEXT:   cuMemHostGetDevicePointer(pDev /*CUdeviceptr **/, pHost /*void **/,
// CUMEMHOSTGETDEVICEPOINTER-NEXT:                             u /*unsigned int*/);
// CUMEMHOSTGETDEVICEPOINTER-NEXT: Is migrated to:
// CUMEMHOSTGETDEVICEPOINTER-NEXT:   *pDev = (dpct::device_ptr)pHost;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostGetFlags | FileCheck %s -check-prefix=CUMEMHOSTGETFLAGS
// CUMEMHOSTGETFLAGS: CUDA API:
// CUMEMHOSTGETFLAGS-NEXT:   cuMemHostGetFlags(pu /*unsigned int **/, pHost /*void **/);
// CUMEMHOSTGETFLAGS-NEXT: Is migrated to:
// CUMEMHOSTGETFLAGS-NEXT:   *pu = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostRegister | FileCheck %s -check-prefix=CUMEMHOSTREGISTER
// CUMEMHOSTREGISTER: CUDA API:
// CUMEMHOSTREGISTER-NEXT:   cuMemHostRegister(pHost /*void **/, s /*size_t*/, u /*unsigned int*/);
// CUMEMHOSTREGISTER-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostUnregister | FileCheck %s -check-prefix=CUMEMHOSTUNREGISTER
// CUMEMHOSTUNREGISTER: CUDA API:
// CUMEMHOSTUNREGISTER-NEXT:   cuMemHostUnregister(pHost /*void **/);
// CUMEMHOSTUNREGISTER-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy | FileCheck %s -check-prefix=CUMEMCPY
// CUMEMCPY: CUDA API:
// CUMEMCPY-NEXT:   cuMemcpy(d1 /*CUdeviceptr*/, d2 /*CUdeviceptr*/, s /*size_t*/);
// CUMEMCPY-NEXT: Is migrated to:
// CUMEMCPY-NEXT:   dpct::get_in_order_queue().memcpy(d1, d2, s).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAsync | FileCheck %s -check-prefix=CUMEMCPYASYNC
// CUMEMCPYASYNC: CUDA API:
// CUMEMCPYASYNC-NEXT:   CUstream cs;
// CUMEMCPYASYNC-NEXT:   cuMemcpyAsync(d1 /*CUdeviceptr*/, d2 /*CUdeviceptr*/, s /*size_t*/,
// CUMEMCPYASYNC-NEXT:                 cs /*CUstream*/);
// CUMEMCPYASYNC-NEXT: Is migrated to:
// CUMEMCPYASYNC-NEXT:   dpct::queue_ptr cs;
// CUMEMCPYASYNC-NEXT:   cs->memcpy(d1, d2, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAtoA | FileCheck %s -check-prefix=CUMEMCPYATOA
// CUMEMCPYATOA: CUDA API:
// CUMEMCPYATOA-NEXT:   CUarray a1;
// CUMEMCPYATOA-NEXT:   CUarray a2;
// CUMEMCPYATOA-NEXT:   cuMemcpyAtoA(a1 /*CUarray*/, s1 /*size_t*/, a2 /*CUarray*/, s2 /*size_t*/,
// CUMEMCPYATOA-NEXT:                s3 /*size_t*/);
// CUMEMCPYATOA-NEXT: Is migrated to:
// CUMEMCPYATOA-NEXT:   dpct::image_matrix_p a1;
// CUMEMCPYATOA-NEXT:   dpct::image_matrix_p a2;
// CUMEMCPYATOA-NEXT:   dpct::dpct_memcpy((char *)(a1->to_pitched_data().get_data_ptr()) + s1, (char *)(a2->to_pitched_data().get_data_ptr()) + s2, s3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAtoD | FileCheck %s -check-prefix=CUMEMCPYATOD
// CUMEMCPYATOD: CUDA API:
// CUMEMCPYATOD-NEXT:   CUarray a;
// CUMEMCPYATOD-NEXT:   cuMemcpyAtoD(d /*CUdeviceptr*/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/);
// CUMEMCPYATOD-NEXT: Is migrated to:
// CUMEMCPYATOD-NEXT:   dpct::image_matrix_p a;
// CUMEMCPYATOD-NEXT:   dpct::dpct_memcpy(d, (char *)(a->to_pitched_data().get_data_ptr()) + s1, s2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAtoH | FileCheck %s -check-prefix=CUMEMCPYATOH
// CUMEMCPYATOH: CUDA API:
// CUMEMCPYATOH-NEXT:   CUarray a;
// CUMEMCPYATOH-NEXT:   cuMemcpyAtoH(pHost /*void **/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/);
// CUMEMCPYATOH-NEXT: Is migrated to:
// CUMEMCPYATOH-NEXT:   dpct::image_matrix_p a;
// CUMEMCPYATOH-NEXT:   dpct::dpct_memcpy(pHost, (char *)(a->to_pitched_data().get_data_ptr()) + s1, s2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAtoHAsync | FileCheck %s -check-prefix=CUMEMCPYATOHASYNC
// CUMEMCPYATOHASYNC: CUDA API:
// CUMEMCPYATOHASYNC-NEXT:   CUarray a;
// CUMEMCPYATOHASYNC-NEXT:   cuMemcpyAtoHAsync(pHost /*void **/, a /*CUarray*/, s1 /*size_t*/,
// CUMEMCPYATOHASYNC-NEXT:                     s2 /*size_t*/, s /*CUstream*/);
// CUMEMCPYATOHASYNC-NEXT: Is migrated to:
// CUMEMCPYATOHASYNC-NEXT:   dpct::image_matrix_p a;
// CUMEMCPYATOHASYNC-NEXT:   dpct::async_dpct_memcpy(pHost, (char *)(a->to_pitched_data().get_data_ptr()) + s1, s2, dpct::automatic, *s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoA | FileCheck %s -check-prefix=CUMEMCPYDTOA
// CUMEMCPYDTOA: CUDA API:
// CUMEMCPYDTOA-NEXT:   CUarray a;
// CUMEMCPYDTOA-NEXT:   cuMemcpyDtoA(a /*CUarray*/, s1 /*size_t*/, d /*CUdeviceptr*/, s2 /*size_t*/);
// CUMEMCPYDTOA-NEXT: Is migrated to:
// CUMEMCPYDTOA-NEXT:   dpct::image_matrix_p a;
// CUMEMCPYDTOA-NEXT:   dpct::dpct_memcpy((char *)(a->to_pitched_data().get_data_ptr()) + s1, d, s2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoD | FileCheck %s -check-prefix=CUMEMCPYDTOD
// CUMEMCPYDTOD: CUDA API:
// CUMEMCPYDTOD-NEXT:   cuMemcpyDtoD(pd1 /*CUdeviceptr*/, pd2 /*CUdeviceptr*/, s /*size_t*/);
// CUMEMCPYDTOD-NEXT: Is migrated to:
// CUMEMCPYDTOD-NEXT:   dpct::get_in_order_queue().memcpy(pd1, pd2, s).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoDAsync | FileCheck %s -check-prefix=CUMEMCPYDTODASYNC
// CUMEMCPYDTODASYNC: CUDA API:
// CUMEMCPYDTODASYNC-NEXT:   CUstream cs;
// CUMEMCPYDTODASYNC-NEXT:   cuMemcpyDtoDAsync(pd1 /*CUdeviceptr*/, pd2 /*CUdeviceptr*/, s /*size_t*/,
// CUMEMCPYDTODASYNC-NEXT:                     cs /*CUstream*/);
// CUMEMCPYDTODASYNC-NEXT: Is migrated to:
// CUMEMCPYDTODASYNC-NEXT:   dpct::queue_ptr cs;
// CUMEMCPYDTODASYNC-NEXT:   cs->memcpy(pd1, pd2, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoH | FileCheck %s -check-prefix=CUMEMCPYDTOH
// CUMEMCPYDTOH: CUDA API:
// CUMEMCPYDTOH-NEXT:   CUdeviceptr pDev;
// CUMEMCPYDTOH-NEXT:   cuMemcpyDtoH(pHost /*void **/, pDev, s /*size_t*/);
// CUMEMCPYDTOH-NEXT: Is migrated to:
// CUMEMCPYDTOH-NEXT:   dpct::device_ptr pDev;
// CUMEMCPYDTOH-NEXT:   dpct::get_in_order_queue().memcpy(pHost /*void **/, pDev, s /*size_t*/).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoHAsync | FileCheck %s -check-prefix=CUMEMCPYDTOHASYNC
// CUMEMCPYDTOHASYNC: CUDA API:
// CUMEMCPYDTOHASYNC-NEXT:   CUdeviceptr pDev;
// CUMEMCPYDTOHASYNC-NEXT:   CUstream stream;
// CUMEMCPYDTOHASYNC-NEXT:   cuMemcpyDtoHAsync(pHost /*void **/, pDev, s /*size_t*/, stream);
// CUMEMCPYDTOHASYNC-NEXT: Is migrated to:
// CUMEMCPYDTOHASYNC-NEXT:   dpct::device_ptr pDev;
// CUMEMCPYDTOHASYNC-NEXT:   dpct::queue_ptr stream;
// CUMEMCPYDTOHASYNC-NEXT:   stream->memcpy(pHost /*void **/, pDev, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyHtoA | FileCheck %s -check-prefix=CUMEMCPYHTOA
// CUMEMCPYHTOA: CUDA API:
// CUMEMCPYHTOA-NEXT:   CUarray a;
// CUMEMCPYHTOA-NEXT:   cuMemcpyHtoA(a /*CUarray*/, s1 /*size_t*/, pHost /*const void **/,
// CUMEMCPYHTOA-NEXT:                s2 /*size_t*/);
// CUMEMCPYHTOA-NEXT: Is migrated to:
// CUMEMCPYHTOA-NEXT:   dpct::image_matrix_p a;
// CUMEMCPYHTOA-NEXT:   dpct::dpct_memcpy((char *)(a->to_pitched_data().get_data_ptr()) + s1, pHost, s2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyHtoAAsync | FileCheck %s -check-prefix=CUMEMCPYHTOAASYNC
// CUMEMCPYHTOAASYNC: CUDA API:
// CUMEMCPYHTOAASYNC-NEXT:   CUarray a;
// CUMEMCPYHTOAASYNC-NEXT:   cuMemcpyHtoAAsync(a /*CUarray*/, s1 /*size_t*/, pHost /*const void **/,
// CUMEMCPYHTOAASYNC-NEXT:                     s2 /*size_t*/, s /*CUstream*/);
// CUMEMCPYHTOAASYNC-NEXT: Is migrated to:
// CUMEMCPYHTOAASYNC-NEXT:   dpct::image_matrix_p a;
// CUMEMCPYHTOAASYNC-NEXT:   dpct::async_dpct_memcpy((char *)(a->to_pitched_data().get_data_ptr()) + s1, pHost, s2, dpct::automatic, *s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyHtoD | FileCheck %s -check-prefix=CUMEMCPYHTOD
// CUMEMCPYHTOD: CUDA API:
// CUMEMCPYHTOD-NEXT:   cuMemcpyHtoD(pDev /*CUdeviceptr*/, pHost /*const void **/, s /*size_t*/);
// CUMEMCPYHTOD-NEXT: Is migrated to:
// CUMEMCPYHTOD-NEXT:   dpct::get_in_order_queue().memcpy(pDev, pHost, s).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyHtoDAsync | FileCheck %s -check-prefix=CUMEMCPYHTODASYNC
// CUMEMCPYHTODASYNC: CUDA API:
// CUMEMCPYHTODASYNC-NEXT:   CUstream stream;
// CUMEMCPYHTODASYNC-NEXT:   cuMemcpyHtoDAsync(pDev /*CUdeviceptr*/, pHost /*const void **/, s /*size_t*/,
// CUMEMCPYHTODASYNC-NEXT:                     stream /*CUstream*/);
// CUMEMCPYHTODASYNC-NEXT: Is migrated to:
// CUMEMCPYHTODASYNC-NEXT:   dpct::queue_ptr stream;
// CUMEMCPYHTODASYNC-NEXT:   stream->memcpy(pDev, pHost, s);

/// Unified Addressing

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAdvise | FileCheck %s -check-prefix=CUMEMADVISE
// CUMEMADVISE: CUDA API:
// CUMEMADVISE-NEXT:   cuMemAdvise(pd /*CUdeviceptr*/, s /*size_t*/, m /*CUmem_advise*/,
// CUMEMADVISE-NEXT:               d /*CUdevice*/);
// CUMEMADVISE-NEXT: Is migrated to:
// CUMEMADVISE-NEXT:   dpct::dev_mgr::instance().get_device(d).in_order_queue().mem_advise(pd, s, m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemPrefetchAsync | FileCheck %s -check-prefix=CUMEMPREFETCHASYNC
// CUMEMPREFETCHASYNC: CUDA API:
// CUMEMPREFETCHASYNC-NEXT:   CUstream cs;
// CUMEMPREFETCHASYNC-NEXT:   cuMemPrefetchAsync(pd /*CUdeviceptr*/, s /*size_t*/, d /*CUdevice*/,
// CUMEMPREFETCHASYNC-NEXT:                      cs /*CUstream*/);
// CUMEMPREFETCHASYNC-NEXT: Is migrated to:
// CUMEMPREFETCHASYNC-NEXT:   dpct::queue_ptr cs;
// CUMEMPREFETCHASYNC-NEXT:   cs->prefetch(pd, s);

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
// CUEVENTQUERY-NEXT:   (int)e->get_info<sycl::info::event::command_execution_status>();

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuLaunchKernel | FileCheck %s -check-prefix=CULAUNCHKERNEL
// CULAUNCHKERNEL: CUDA API:
// CULAUNCHKERNEL-NEXT:   cuLaunchKernel(f /*CUfunction*/, u1 /*unsigned int*/, u2 /*unsigned int*/,
// CULAUNCHKERNEL-NEXT:                  u3 /*unsigned int*/, u4 /*unsigned int*/, u5 /*unsigned int*/,
// CULAUNCHKERNEL-NEXT:                  u6 /*unsigned int*/, u7 /*unsigned int*/, s /*CUstream*/,
// CULAUNCHKERNEL-NEXT:                  pParam /*void ***/, pOpt /*void ***/);
// CULAUNCHKERNEL-NEXT: Is migrated to:
// CULAUNCHKERNEL-NEXT:   dpct::invoke_kernel_function(f, *s, sycl::range<3>(u3, u2, u1), sycl::range<3>(u6, u5, u4), u7, pParam, pOpt);

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
// CUTEXREFSETADDRESS2D-NEXT:   cuTexRefSetAddress2D(t /*CUtexref*/, pa /*size_t **/, d /*CUdeviceptr*/,
// CUTEXREFSETADDRESS2D-NEXT:                        s /*size_t*/);
// CUTEXREFSETADDRESS2D-NEXT: Is migrated to:
// CUTEXREFSETADDRESS2D-NEXT:   dpct::image_wrapper_base_p t;
// CUTEXREFSETADDRESS2D-NEXT:   t->attach(d, s);

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
