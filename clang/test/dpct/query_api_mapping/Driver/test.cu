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
// CUDRIVERGETVERSION-EMPTY:

/// Device Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGet | FileCheck %s -check-prefix=CUDEVICEGET
// CUDEVICEGET: CUDA API:
// CUDEVICEGET-NEXT:   cuDeviceGet(pd /*CUdevice **/, i /*int*/);
// CUDEVICEGET-NEXT: Is migrated to:
// CUDEVICEGET-NEXT:   *pd = i;
// CUDEVICEGET-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetCount | FileCheck %s -check-prefix=CUDEVICEGETCOUNT
// CUDEVICEGETCOUNT: CUDA API:
// CUDEVICEGETCOUNT-NEXT:   cuDeviceGetCount(pi /*int **/);
// CUDEVICEGETCOUNT-NEXT: Is migrated to:
// CUDEVICEGETCOUNT-NEXT:   *pi = dpct::dev_mgr::instance().device_count();
// CUDEVICEGETCOUNT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceGetName | FileCheck %s -check-prefix=CUDEVICEGETNAME
// CUDEVICEGETNAME: CUDA API:
// CUDEVICEGETNAME-NEXT:   cuDeviceGetName(pc /*char **/, i /*int*/, d /*CUdevice*/);
// CUDEVICEGETNAME-NEXT: Is migrated to:
// CUDEVICEGETNAME-NEXT:   memcpy(pc, dpct::dev_mgr::instance().get_device(d).get_info<sycl::info::device::name>().c_str(), i);
// CUDEVICEGETNAME-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceTotalMem | FileCheck %s -check-prefix=CUDEVICETOTALMEM
// CUDEVICETOTALMEM: CUDA API:
// CUDEVICETOTALMEM-NEXT:   cuDeviceTotalMem(ps /*size_t **/, d /*CUdevice*/);
// CUDEVICETOTALMEM-NEXT: Is migrated to:
// CUDEVICETOTALMEM-NEXT:   *ps = dpct::dev_mgr::instance().get_device(d).get_device_info().get_global_mem_size();
// CUDEVICETOTALMEM-EMPTY:

/// Device Management [DEPRECATED]

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuDeviceComputeCapability | FileCheck %s -check-prefix=CUDEVICECOMPUTECAPABILITY
// CUDEVICECOMPUTECAPABILITY: CUDA API:
// CUDEVICECOMPUTECAPABILITY-NEXT:   cuDeviceComputeCapability(pi1 /*int **/, pi2 /*int **/, d /*CUdevice*/);
// CUDEVICECOMPUTECAPABILITY-NEXT: Is migrated to:
// CUDEVICECOMPUTECAPABILITY-NEXT:   *pi1 = dpct::dev_mgr::instance().get_device(d).get_major_version();
// CUDEVICECOMPUTECAPABILITY-NEXT:   *pi2 = dpct::dev_mgr::instance().get_device(d).get_minor_version();
// CUDEVICECOMPUTECAPABILITY-EMPTY:

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
// CUDEVICEPRIMARYCTXRETAIN-EMPTY:

/// Context Management

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
// CUCTXGETAPIVERSION-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxGetCurrent | FileCheck %s -check-prefix=CUCTXGETCURRENT
// CUCTXGETCURRENT: CUDA API:
// CUCTXGETCURRENT-NEXT:   cuCtxGetCurrent(pc /*CUcontext **/);
// CUCTXGETCURRENT-NEXT: Is migrated to:
// CUCTXGETCURRENT-NEXT:   *pc = dpct::dev_mgr::instance().current_device_id();
// CUCTXGETCURRENT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxGetDevice | FileCheck %s -check-prefix=CUCTXGETDEVICE
// CUCTXGETDEVICE: CUDA API:
// CUCTXGETDEVICE-NEXT:   cuCtxGetDevice(pd /*CUdevice **/);
// CUCTXGETDEVICE-NEXT: Is migrated to:
// CUCTXGETDEVICE-NEXT:   *pd = dpct::get_current_device_id();
// CUCTXGETDEVICE-EMPTY:

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
// CUCTXSETCURRENT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCtxSynchronize | FileCheck %s -check-prefix=CUCTXSYNCHRONIZE
// CUCTXSYNCHRONIZE: CUDA API:
// CUCTXSYNCHRONIZE-NEXT:   cuCtxSynchronize();
// CUCTXSYNCHRONIZE-NEXT: Is migrated to:
// CUCTXSYNCHRONIZE-NEXT:   dpct::get_current_device().queues_wait_and_throw();
// CUCTXSYNCHRONIZE-EMPTY:

/// Module Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleGetFunction | FileCheck %s -check-prefix=CUMODULEGETFUNCTION
// CUMODULEGETFUNCTION: CUDA API:
// CUMODULEGETFUNCTION-NEXT:   cuModuleGetFunction(pf /*CUfunction **/, m /*CUmodule*/, pc /*const char **/);
// CUMODULEGETFUNCTION-NEXT: Is migrated to:
// CUMODULEGETFUNCTION-NEXT:   *pf = dpct::get_kernel_function(m, pc);
// CUMODULEGETFUNCTION-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleLoad | FileCheck %s -check-prefix=CUMODULELOAD
// CUMODULELOAD: CUDA API:
// CUMODULELOAD-NEXT:   cuModuleLoad(pm /*CUmodule **/, pc /*const char **/);
// CUMODULELOAD-NEXT: Is migrated to:
// CUMODULELOAD-NEXT:   *pm = dpct::load_kernel_library(pc);
// CUMODULELOAD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleLoadData | FileCheck %s -check-prefix=CUMODULELOADDATA
// CUMODULELOADDATA: CUDA API:
// CUMODULELOADDATA-NEXT:   cuModuleLoadData(pm /*CUmodule **/, pv /*const void **/);
// CUMODULELOADDATA-NEXT: Is migrated to:
// CUMODULELOADDATA-NEXT:    *pm = dpct::load_kernel_library_mem(pv);
// CUMODULELOADDATA-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleLoadDataEx | FileCheck %s -check-prefix=CUMODULELOADDATAEX
// CUMODULELOADDATAEX: CUDA API:
// CUMODULELOADDATAEX-NEXT:   cuModuleLoadDataEx(pm /*CUmodule **/, pv /*const void **/, u /*unsigned int*/,
// CUMODULELOADDATAEX-NEXT:                      pj /*CUjit_option **/, ppv /*void ***/);
// CUMODULELOADDATAEX-NEXT: Is migrated to:
// CUMODULELOADDATAEX-NEXT:   *pm = dpct::load_kernel_library_mem(pv);
// CUMODULELOADDATAEX-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleUnload | FileCheck %s -check-prefix=CUMODULEUNLOAD
// CUMODULEUNLOAD: CUDA API:
// CUMODULEUNLOAD-NEXT:   cuModuleUnload(m /*CUmodule*/);
// CUMODULEUNLOAD-NEXT: Is migrated to:
// CUMODULEUNLOAD-NEXT:   dpct::unload_kernel_library(m);
// CUMODULEUNLOAD-EMPTY:

/// Module Management [DEPRECATED]

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuModuleGetTexRef | FileCheck %s -check-prefix=CUMODULEGETTEXREF
// CUMODULEGETTEXREF: CUDA API:
// CUMODULEGETTEXREF-NEXT:   cuModuleGetTexRef(pt /*CUtexref **/, m /*CUmodule*/, pc /*const char **/);
// CUMODULEGETTEXREF-NEXT: Is migrated to:
// CUMODULEGETTEXREF-NEXT:   *pt = dpct::get_image_wrapper(m, pc);
// CUMODULEGETTEXREF-EMPTY:

/// Memory Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuArrayDestroy | FileCheck %s -check-prefix=CUARRAYDESTROY
// CUARRAYDESTROY: CUDA API:
// CUARRAYDESTROY-NEXT:   cuArrayDestroy(a /*CUarray*/);
// CUARRAYDESTROY-NEXT: Is migrated to:
// CUARRAYDESTROY-NEXT:   delete a;
// CUARRAYDESTROY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAlloc | FileCheck %s -check-prefix=CUMEMALLOC
// CUMEMALLOC: CUDA API:
// CUMEMALLOC-NEXT:   cuMemAlloc(pd /*CUdeviceptr **/, s /*size_t*/);
// CUMEMALLOC-NEXT: Is migrated to:
// CUMEMALLOC-NEXT:   *pd = (dpct::device_ptr)sycl::malloc_device(s, dpct::get_default_queue());
// CUMEMALLOC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAllocHost | FileCheck %s -check-prefix=CUMEMALLOCHOST
// CUMEMALLOCHOST: CUDA API:
// CUMEMALLOCHOST-NEXT:   cuMemAllocHost(ppv /*void ***/, s /*size_t*/);
// CUMEMALLOCHOST-NEXT: Is migrated to:
// CUMEMALLOCHOST-NEXT:   *ppv = (void *)sycl::malloc_host(s, dpct::get_default_queue());
// CUMEMALLOCHOST-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAllocManaged | FileCheck %s -check-prefix=CUMEMALLOCMANAGED
// CUMEMALLOCMANAGED: CUDA API:
// CUMEMALLOCMANAGED-NEXT:   cuMemAllocManaged(pd /*CUdeviceptr **/, s /*size_t*/, u /*unsigned int*/);
// CUMEMALLOCMANAGED-NEXT: Is migrated to:
// CUMEMALLOCMANAGED-NEXT:   *pd = (dpct::device_ptr)sycl::malloc_shared(s, dpct::get_default_queue());
// CUMEMALLOCMANAGED-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAllocPitch | FileCheck %s -check-prefix=CUMEMALLOCPITCH
// CUMEMALLOCPITCH: CUDA API:
// CUMEMALLOCPITCH-NEXT:   cuMemAllocPitch(pd /*CUdeviceptr **/, ps /*size_t **/, s1 /*size_t*/,
// CUMEMALLOCPITCH-NEXT:                   s2 /*size_t*/, u /*unsigned int*/);
// CUMEMALLOCPITCH-NEXT: Is migrated to:
// CUMEMALLOCPITCH-NEXT:   *pd = (dpct::device_ptr)dpct::dpct_malloc(*ps, s1, s2);
// CUMEMALLOCPITCH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemFree | FileCheck %s -check-prefix=CUMEMFREE
// CUMEMFREE: CUDA API:
// CUMEMFREE-NEXT:   cuMemFree(d /*CUdeviceptr*/);
// CUMEMFREE-NEXT: Is migrated to:
// CUMEMFREE-NEXT:   sycl::free(d, dpct::get_default_queue());
// CUMEMFREE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemFreeHost | FileCheck %s -check-prefix=CUMEMFREEHOST
// CUMEMFREEHOST: CUDA API:
// CUMEMFREEHOST-NEXT:   cuMemFreeHost(pv /*void **/);
// CUMEMFREEHOST-NEXT: Is migrated to:
// CUMEMFREEHOST-NEXT:   sycl::free(pv, dpct::get_default_queue());
// CUMEMFREEHOST-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemGetInfo | FileCheck %s -check-prefix=CUMEMGETINFO
// CUMEMGETINFO: CUDA API:
// CUMEMGETINFO-NEXT:   cuMemGetInfo(ps1 /*size_t **/, ps2 /*size_t **/);
// CUMEMGETINFO-NEXT: Is migrated to:
// CUMEMGETINFO-NEXT:   dpct::get_current_device().get_memory_info(*ps1, *ps2);
// CUMEMGETINFO-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostAlloc | FileCheck %s -check-prefix=CUMEMHOSTALLOC
// CUMEMHOSTALLOC: CUDA API:
// CUMEMHOSTALLOC-NEXT:   cuMemHostAlloc(ppv /*void ***/, s /*size_t*/, u /*unsigned int*/);
// CUMEMHOSTALLOC-NEXT: Is migrated to:
// CUMEMHOSTALLOC-NEXT:   *ppv = (void *)sycl::malloc_host(s, dpct::get_default_queue());
// CUMEMHOSTALLOC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostGetDevicePointer | FileCheck %s -check-prefix=CUMEMHOSTGETDEVICEPOINTER
// CUMEMHOSTGETDEVICEPOINTER: CUDA API:
// CUMEMHOSTGETDEVICEPOINTER-NEXT:   cuMemHostGetDevicePointer(pd /*CUdeviceptr **/, pv /*void **/,
// CUMEMHOSTGETDEVICEPOINTER-NEXT:                             u /*unsigned int*/);
// CUMEMHOSTGETDEVICEPOINTER-NEXT: Is migrated to:
// CUMEMHOSTGETDEVICEPOINTER-NEXT:   *pd = (dpct::device_ptr)pv;
// CUMEMHOSTGETDEVICEPOINTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostGetFlags | FileCheck %s -check-prefix=CUMEMHOSTGETFLAGS
// CUMEMHOSTGETFLAGS: CUDA API:
// CUMEMHOSTGETFLAGS-NEXT:   cuMemHostGetFlags(pu /*unsigned int **/, pv /*void **/);
// CUMEMHOSTGETFLAGS-NEXT: Is migrated to:
// CUMEMHOSTGETFLAGS-NEXT:   *pu = 0;
// CUMEMHOSTGETFLAGS-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostRegister | FileCheck %s -check-prefix=CUMEMHOSTREGISTER
// CUMEMHOSTREGISTER: CUDA API:
// CUMEMHOSTREGISTER-NEXT:   cuMemHostRegister(pv /*void **/, s /*size_t*/, u /*unsigned int*/);
// CUMEMHOSTREGISTER-NEXT: The API is Removed.
// CUMEMHOSTREGISTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostUnregister | FileCheck %s -check-prefix=CUMEMHOSTUNREGISTER
// CUMEMHOSTUNREGISTER: CUDA API:
// CUMEMHOSTUNREGISTER-NEXT:   cuMemHostUnregister(pv /*void **/);
// CUMEMHOSTUNREGISTER-NEXT: The API is Removed.
// CUMEMHOSTUNREGISTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy | FileCheck %s -check-prefix=CUMEMCPY
// CUMEMCPY: CUDA API:
// CUMEMCPY-NEXT:   cuMemcpy(d1 /*CUdeviceptr*/, d2 /*CUdeviceptr*/, s /*size_t*/);
// CUMEMCPY-NEXT: Is migrated to:
// CUMEMCPY-NEXT:   dpct::get_default_queue().memcpy(d1, d2, s).wait();
// CUMEMCPY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy2D | FileCheck %s -check-prefix=CUMEMCPY2D
// CUMEMCPY2D: CUDA API:
// CUMEMCPY2D-NEXT:   cuMemcpy2D(pm /*const CUDA_MEMCPY2D **/);
// CUMEMCPY2D-NEXT: Is migrated to:
// CUMEMCPY2D-NEXT:   dpct::dpct_memcpy(pm /*const CUDA_MEMCPY2D **/);
// CUMEMCPY2D-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy2DAsync | FileCheck %s -check-prefix=CUMEMCPY2DASYNC
// CUMEMCPY2DASYNC: CUDA API:
// CUMEMCPY2DASYNC-NEXT:   cuMemcpy2DAsync(pm /*const CUDA_MEMCPY2D **/, s /*CUstream*/);
// CUMEMCPY2DASYNC-NEXT: Is migrated to:
// CUMEMCPY2DASYNC-NEXT:   dpct::async_dpct_memcpy(pm /*const CUDA_MEMCPY2D **/, *s /*CUstream*/);
// CUMEMCPY2DASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy3D | FileCheck %s -check-prefix=CUMEMCPY3D
// CUMEMCPY3D: CUDA API:
// CUMEMCPY3D-NEXT:   cuMemcpy3D(pm /*const CUDA_MEMCPY3D **/);
// CUMEMCPY3D-NEXT: Is migrated to:
// CUMEMCPY3D-NEXT:   dpct::dpct_memcpy(pm /*const CUDA_MEMCPY3D **/);
// CUMEMCPY3D-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy3DAsync | FileCheck %s -check-prefix=CUMEMCPY3DASYNC
// CUMEMCPY3DASYNC: CUDA API:
// CUMEMCPY3DASYNC-NEXT:   cuMemcpy3DAsync(pm /*const CUDA_MEMCPY3D **/, s /*CUstream*/);
// CUMEMCPY3DASYNC-NEXT: Is migrated to:
// CUMEMCPY3DASYNC-NEXT:   dpct::async_dpct_memcpy(pm /*const CUDA_MEMCPY3D **/, *s /*CUstream*/);
// CUMEMCPY3DASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAsync | FileCheck %s -check-prefix=CUMEMCPYASYNC
// CUMEMCPYASYNC: CUDA API:
// CUMEMCPYASYNC-NEXT:   cuMemcpyAsync(d1 /*CUdeviceptr*/, d2 /*CUdeviceptr*/, s /*size_t*/,
// CUMEMCPYASYNC-NEXT:                 cs /*CUstream*/);
// CUMEMCPYASYNC-NEXT: Is migrated to:
// CUMEMCPYASYNC-NEXT:   cs->memcpy(d1, d2, s);
// CUMEMCPYASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAtoA | FileCheck %s -check-prefix=CUMEMCPYATOA
// CUMEMCPYATOA: CUDA API:
// CUMEMCPYATOA-NEXT:   cuMemcpyAtoA(a1 /*CUarray*/, s1 /*size_t*/, a2 /*CUarray*/, s2 /*size_t*/,
// CUMEMCPYATOA-NEXT:                s3 /*size_t*/);
// CUMEMCPYATOA-NEXT: Is migrated to:
// CUMEMCPYATOA-NEXT:   dpct::dpct_memcpy((char *)(a1->to_pitched_data().get_data_ptr()) + s1, (char *)(a2->to_pitched_data().get_data_ptr()) + s2, s3);
// CUMEMCPYATOA-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAtoD | FileCheck %s -check-prefix=CUMEMCPYATOD
// CUMEMCPYATOD: CUDA API:
// CUMEMCPYATOD-NEXT:   cuMemcpyAtoD(d /*CUdeviceptr*/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/);
// CUMEMCPYATOD-NEXT: Is migrated to:
// CUMEMCPYATOD-NEXT:   dpct::dpct_memcpy(d, (char *)(a->to_pitched_data().get_data_ptr()) + s1, s2);
// CUMEMCPYATOD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAtoH | FileCheck %s -check-prefix=CUMEMCPYATOH
// CUMEMCPYATOH: CUDA API:
// CUMEMCPYATOH-NEXT:   cuMemcpyAtoH(pv /*void **/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/);
// CUMEMCPYATOH-NEXT: Is migrated to:
// CUMEMCPYATOH-NEXT:   dpct::dpct_memcpy(pv, (char *)(a->to_pitched_data().get_data_ptr()) + s1, s2);
// CUMEMCPYATOH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyAtoHAsync | FileCheck %s -check-prefix=CUMEMCPYATOHASYNC
// CUMEMCPYATOHASYNC: CUDA API:
// CUMEMCPYATOHASYNC-NEXT:   cuMemcpyAtoHAsync(pv /*void **/, a /*CUarray*/, s1 /*size_t*/, s2 /*size_t*/,
// CUMEMCPYATOHASYNC-NEXT:                     s /*CUstream*/);
// CUMEMCPYATOHASYNC-NEXT: Is migrated to:
// CUMEMCPYATOHASYNC-NEXT:   dpct::async_dpct_memcpy(pv, (char *)(a->to_pitched_data().get_data_ptr()) + s1, s2, dpct::automatic, *s);
// CUMEMCPYATOHASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoA | FileCheck %s -check-prefix=CUMEMCPYDTOA
// CUMEMCPYDTOA: CUDA API:
// CUMEMCPYDTOA-NEXT:   cuMemcpyDtoA(a /*CUarray*/, s1 /*size_t*/, d /*CUdeviceptr*/, s2 /*size_t*/);
// CUMEMCPYDTOA-NEXT: Is migrated to:
// CUMEMCPYDTOA-NEXT:   dpct::dpct_memcpy((char *)(a->to_pitched_data().get_data_ptr()) + s1, d, s2);
// CUMEMCPYDTOA-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoD | FileCheck %s -check-prefix=CUMEMCPYDTOD
// CUMEMCPYDTOD: CUDA API:
// CUMEMCPYDTOD-NEXT:   cuMemcpyDtoD(pd1 /*CUdeviceptr*/, pd2 /*CUdeviceptr*/, s /*size_t*/);
// CUMEMCPYDTOD-NEXT: Is migrated to:
// CUMEMCPYDTOD-NEXT:   dpct::get_default_queue().memcpy(pd1, pd2, s).wait();
// CUMEMCPYDTOD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoDAsync | FileCheck %s -check-prefix=CUMEMCPYDTODASYNC
// CUMEMCPYDTODASYNC: CUDA API:
// CUMEMCPYDTODASYNC-NEXT:   cuMemcpyDtoDAsync(pd1 /*CUdeviceptr*/, pd2 /*CUdeviceptr*/, s /*size_t*/,
// CUMEMCPYDTODASYNC-NEXT:                     cs /*CUstream*/);
// CUMEMCPYDTODASYNC-NEXT: Is migrated to:
// CUMEMCPYDTODASYNC-NEXT:   cs->memcpy(pd1, pd2, s);
// CUMEMCPYDTODASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoH | FileCheck %s -check-prefix=CUMEMCPYDTOH
// CUMEMCPYDTOH: CUDA API:
// CUMEMCPYDTOH-NEXT:   cuMemcpyDtoH(pv /*void **/, pd /*CUdeviceptr*/, s /*size_t*/);
// CUMEMCPYDTOH-NEXT: Is migrated to:
// CUMEMCPYDTOH-NEXT:   dpct::get_default_queue().memcpy(pv /*void **/, pd /*CUdeviceptr*/, s /*size_t*/).wait();
// CUMEMCPYDTOH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyDtoHAsync | FileCheck %s -check-prefix=CUMEMCPYDTOHASYNC
// CUMEMCPYDTOHASYNC: CUDA API:
// CUMEMCPYDTOHASYNC-NEXT:   cuMemcpyDtoHAsync(pv /*void **/, pd /*CUdeviceptr*/, s /*size_t*/,
// CUMEMCPYDTOHASYNC-NEXT:                     cs /*CUstream*/);
// CUMEMCPYDTOHASYNC-NEXT: Is migrated to:
// CUMEMCPYDTOHASYNC-NEXT:   cs->memcpy(pv /*void **/, pd /*CUdeviceptr*/, s /*CUstream*/);
// CUMEMCPYDTOHASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyHtoA | FileCheck %s -check-prefix=CUMEMCPYHTOA
// CUMEMCPYHTOA: CUDA API:
// CUMEMCPYHTOA-NEXT:   cuMemcpyHtoA(a /*CUarray*/, s1 /*size_t*/, pv /*const void **/,
// CUMEMCPYHTOA-NEXT:                s2 /*size_t*/);
// CUMEMCPYHTOA-NEXT: Is migrated to:
// CUMEMCPYHTOA-NEXT:   dpct::dpct_memcpy((char *)(a->to_pitched_data().get_data_ptr()) + s1, pv, s2);
// CUMEMCPYHTOA-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyHtoAAsync | FileCheck %s -check-prefix=CUMEMCPYHTOAASYNC
// CUMEMCPYHTOAASYNC: CUDA API:
// CUMEMCPYHTOAASYNC-NEXT:   cuMemcpyHtoAAsync(a /*CUarray*/, s1 /*size_t*/, pv /*const void **/,
// CUMEMCPYHTOAASYNC-NEXT:                     s2 /*size_t*/, s /*CUstream*/);
// CUMEMCPYHTOAASYNC-NEXT: Is migrated to:
// CUMEMCPYHTOAASYNC-NEXT:   dpct::async_dpct_memcpy((char *)(a->to_pitched_data().get_data_ptr()) + s1, pv, s2, dpct::automatic, *s);
// CUMEMCPYHTOAASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyHtoD | FileCheck %s -check-prefix=CUMEMCPYHTOD
// CUMEMCPYHTOD: CUDA API:
// CUMEMCPYHTOD-NEXT:   cuMemcpyHtoD(pd /*CUdeviceptr*/, pv /*const void **/, s /*size_t*/);
// CUMEMCPYHTOD-NEXT: Is migrated to:
// CUMEMCPYHTOD-NEXT:   dpct::get_default_queue().memcpy(pd, pv, s).wait();
// CUMEMCPYHTOD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyHtoDAsync | FileCheck %s -check-prefix=CUMEMCPYHTODASYNC
// CUMEMCPYHTODASYNC: CUDA API:
// CUMEMCPYHTODASYNC-NEXT:   cuMemcpyHtoDAsync(pd /*CUdeviceptr*/, pv /*const void **/, s /*size_t*/,
// CUMEMCPYHTODASYNC-NEXT:                     cs /*CUstream*/);
// CUMEMCPYHTODASYNC-NEXT: Is migrated to:
// CUMEMCPYHTODASYNC-NEXT:   cs->memcpy(pd, pv, s);
// CUMEMCPYHTODASYNC-EMPTY:

/// Unified Addressing

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemAdvise | FileCheck %s -check-prefix=CUMEMADVISE
// CUMEMADVISE: CUDA API:
// CUMEMADVISE-NEXT:   cuMemAdvise(pd /*CUdeviceptr*/, s /*size_t*/, m /*CUmem_advise*/,
// CUMEMADVISE-NEXT:               d /*CUdevice*/);
// CUMEMADVISE-NEXT: Is migrated to:
// CUMEMADVISE-NEXT:   dpct::dev_mgr::instance().get_device(d).default_queue().mem_advise(pd, s, m);
// CUMEMADVISE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemPrefetchAsync | FileCheck %s -check-prefix=CUMEMPREFETCHASYNC
// CUMEMPREFETCHASYNC: CUDA API:
// CUMEMPREFETCHASYNC-NEXT:   cuMemPrefetchAsync(pd /*CUdeviceptr*/, s /*size_t*/, d /*CUdevice*/,
// CUMEMPREFETCHASYNC-NEXT:                      cs /*CUstream*/);
// CUMEMPREFETCHASYNC-NEXT: Is migrated to:
// CUMEMPREFETCHASYNC-NEXT:    cs->prefetch(pd, s);
// CUMEMPREFETCHASYNC-EMPTY:

/// Stream Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamAddCallback | FileCheck %s -check-prefix=CUSTREAMADDCALLBACK
// CUSTREAMADDCALLBACK: CUDA API:
// CUSTREAMADDCALLBACK-NEXT:   cuStreamAddCallback(s /*CUstream*/, sc /*CUstreamCallback*/, pv /*void **/,
// CUSTREAMADDCALLBACK-NEXT:                       u /*unsigned int*/);
// CUSTREAMADDCALLBACK-NEXT: Is migrated to:
// CUSTREAMADDCALLBACK-NEXT:   std::async([&](){s->wait(); sc(s, 0, pv);});
// CUSTREAMADDCALLBACK-EMPTY:

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
// CUSTREAMCREATE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamDestroy | FileCheck %s -check-prefix=CUSTREAMDESTROY
// CUSTREAMDESTROY: CUDA API:
// CUSTREAMDESTROY-NEXT:   cuStreamDestroy(s /*CUstream*/);
// CUSTREAMDESTROY-NEXT: Is migrated to:
// CUSTREAMDESTROY-NEXT:   dpct::get_current_device().destroy_queue(s);
// CUSTREAMDESTROY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamSynchronize | FileCheck %s -check-prefix=CUSTREAMSYNCHRONIZE
// CUSTREAMSYNCHRONIZE: CUDA API:
// CUSTREAMSYNCHRONIZE-NEXT:   cuStreamSynchronize(s /*CUstream*/);
// CUSTREAMSYNCHRONIZE-NEXT: Is migrated to:
// CUSTREAMSYNCHRONIZE-NEXT:   s->wait();
// CUSTREAMSYNCHRONIZE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuStreamWaitEvent | FileCheck %s -check-prefix=CUSTREAMWAITEVENT
// CUSTREAMWAITEVENT: CUDA API:
// CUSTREAMWAITEVENT-NEXT:   cuStreamWaitEvent(s /*CUstream*/, e /*CUevent*/, u /*unsigned int*/);
// CUSTREAMWAITEVENT-NEXT: Is migrated to:
// CUSTREAMWAITEVENT-NEXT:   s->ext_oneapi_submit_barrier({*e});
// CUSTREAMWAITEVENT-EMPTY:

/// Event Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventCreate | FileCheck %s -check-prefix=CUEVENTCREATE
// CUEVENTCREATE: CUDA API:
// CUEVENTCREATE-NEXT:   cuEventCreate(pe /*CUevent **/, u /*unsigned int*/);
// CUEVENTCREATE-NEXT: Is migrated to:
// CUEVENTCREATE-NEXT:   *pe = new sycl::event();
// CUEVENTCREATE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventDestroy | FileCheck %s -check-prefix=CUEVENTDESTROY
// CUEVENTDESTROY: CUDA API:
// CUEVENTDESTROY-NEXT:   cuEventDestroy(e /*CUevent*/);
// CUEVENTDESTROY-NEXT: Is migrated to:
// CUEVENTDESTROY-NEXT:   dpct::destroy_event(e);
// CUEVENTDESTROY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventElapsedTime | FileCheck %s -check-prefix=CUEVENTELAPSEDTIME
// CUEVENTELAPSEDTIME: CUDA API:
// CUEVENTELAPSEDTIME-NEXT:   cuEventElapsedTime(pf /*float **/, e1 /*CUevent*/, e2 /*CUevent*/);
// CUEVENTELAPSEDTIME-NEXT: Is migrated to:
// CUEVENTELAPSEDTIME-NEXT:   *(pf) = std::chrono::duration<float, std::milli>(e2_ct1 - e1_ct1).count();
// CUEVENTELAPSEDTIME-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventQuery | FileCheck %s -check-prefix=CUEVENTQUERY
// CUEVENTQUERY: CUDA API:
// CUEVENTQUERY-NEXT:   cuEventQuery(e /*CUevent*/);
// CUEVENTQUERY-NEXT: Is migrated to:
// CUEVENTQUERY-NEXT:   (int)e->get_info<sycl::info::event::command_execution_status>();
// CUEVENTQUERY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventRecord | FileCheck %s -check-prefix=CUEVENTRECORD
// CUEVENTRECORD: CUDA API:
// CUEVENTRECORD-NEXT:   cuEventRecord(e /*CUevent*/, s /*CUstream*/);
// CUEVENTRECORD-NEXT: Is migrated to:
// CUEVENTRECORD-NEXT:   ;
// CUEVENTRECORD-NEXT:   *e = s->ext_oneapi_submit_barrier();
// CUEVENTRECORD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuEventSynchronize | FileCheck %s -check-prefix=CUEVENTSYNCHRONIZE
// CUEVENTSYNCHRONIZE: CUDA API:
// CUEVENTSYNCHRONIZE-NEXT:   cuEventSynchronize(e /*CUevent*/);
// CUEVENTSYNCHRONIZE-NEXT: Is migrated to:
// CUEVENTSYNCHRONIZE-NEXT:  e->wait_and_throw();
// CUEVENTSYNCHRONIZE-EMPTY:
