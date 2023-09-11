// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

/// Device Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceGetCacheConfig | FileCheck %s -check-prefix=CUDADEVICEGETCACHECONFIG
// CUDADEVICEGETCACHECONFIG: CUDA API:
// CUDADEVICEGETCACHECONFIG-NEXT:   cudaDeviceGetCacheConfig(pf /*enum cudaFuncCache **/);
// CUDADEVICEGETCACHECONFIG-NEXT: The API is Removed.
// CUDADEVICEGETCACHECONFIG-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceGetLimit | FileCheck %s -check-prefix=CUDADEVICEGETLIMIT
// CUDADEVICEGETLIMIT: CUDA API:
// CUDADEVICEGETLIMIT-NEXT:   cudaDeviceGetLimit(ps /*size_t **/, l /*cudaLimit*/);
// CUDADEVICEGETLIMIT-NEXT: Is migrated to:
// CUDADEVICEGETLIMIT-NEXT:   *ps = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceReset | FileCheck %s -check-prefix=CUDADEVICERESET
// CUDADEVICERESET: CUDA API:
// CUDADEVICERESET-NEXT:   cudaDeviceReset();
// CUDADEVICERESET-NEXT: Is migrated to:
// CUDADEVICERESET-NEXT:   dpct::get_current_device().reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceSetCacheConfig | FileCheck %s -check-prefix=CUDADEVICESETCACHECONFIG
// CUDADEVICESETCACHECONFIG: CUDA API:
// CUDADEVICESETCACHECONFIG-NEXT:   cudaDeviceSetCacheConfig(f /*cudaFuncCache*/);
// CUDADEVICESETCACHECONFIG-NEXT: The API is Removed.
// CUDADEVICESETCACHECONFIG-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceSetLimit | FileCheck %s -check-prefix=CUDADEVICESETLIMIT
// CUDADEVICESETLIMIT: CUDA API:
// CUDADEVICESETLIMIT-NEXT:   cudaDeviceSetLimit(l /*cudaLimit*/, s /*size_t*/);
// CUDADEVICESETLIMIT-NEXT: The API is Removed.
// CUDADEVICESETLIMIT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceSetSharedMemConfig | FileCheck %s -check-prefix=CUDADEVICESETSHAREDMEMCONFIG
// CUDADEVICESETSHAREDMEMCONFIG: CUDA API:
// CUDADEVICESETSHAREDMEMCONFIG-NEXT:   cudaDeviceSetSharedMemConfig(s /*cudaSharedMemConfig*/);
// CUDADEVICESETSHAREDMEMCONFIG-NEXT: The API is Removed.
// CUDADEVICESETSHAREDMEMCONFIG-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceSynchronize | FileCheck %s -check-prefix=CUDADEVICESYNCHRONIZE
// CUDADEVICESYNCHRONIZE: CUDA API:
// CUDADEVICESYNCHRONIZE-NEXT:   cudaDeviceSynchronize();
// CUDADEVICESYNCHRONIZE-NEXT: Is migrated to:
// CUDADEVICESYNCHRONIZE-NEXT:   dpct::get_current_device().queues_wait_and_throw();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetDevice | FileCheck %s -check-prefix=CUDAGETDEVICE
// CUDAGETDEVICE: CUDA API:
// CUDAGETDEVICE-NEXT:   cudaGetDevice(pi /*int **/);
// CUDAGETDEVICE-NEXT: Is migrated to:
// CUDAGETDEVICE-NEXT:   *pi = dpct::dev_mgr::instance().current_device_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetDeviceCount | FileCheck %s -check-prefix=CUDAGETDEVICECOUNT
// CUDAGETDEVICECOUNT: CUDA API:
// CUDAGETDEVICECOUNT-NEXT:   cudaGetDeviceCount(i /*int*/);
// CUDAGETDEVICECOUNT-NEXT: Is migrated to:
// CUDAGETDEVICECOUNT-NEXT:   *i = dpct::dev_mgr::instance().device_count();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetDeviceProperties | FileCheck %s -check-prefix=CUDAGETDEVICEPROPERTIES
// CUDAGETDEVICEPROPERTIES: CUDA API:
// CUDAGETDEVICEPROPERTIES-NEXT:   cudaDeviceProp *pd;
// CUDAGETDEVICEPROPERTIES-NEXT:   cudaGetDeviceProperties(pd, i /*int*/);
// CUDAGETDEVICEPROPERTIES-NEXT: Is migrated to:
// CUDAGETDEVICEPROPERTIES-NEXT:   dpct::device_info *pd;
// CUDAGETDEVICEPROPERTIES-NEXT:   dpct::get_device_info(*pd, dpct::dev_mgr::instance().get_device(i) /*int*/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaSetDevice | FileCheck %s -check-prefix=CUDASETDEVICE
// CUDASETDEVICE: CUDA API:
// CUDASETDEVICE-NEXT:   cudaSetDevice(i /*int*/);
// CUDASETDEVICE-NEXT: Is migrated to:
// CUDASETDEVICE-NEXT:   dpct::select_device(i /*int*/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaSetDeviceFlags | FileCheck %s -check-prefix=CUDASETDEVICEFLAGS
// CUDASETDEVICEFLAGS: CUDA API:
// CUDASETDEVICEFLAGS-NEXT:   cudaSetDeviceFlags(u /*unsigned int*/);
// CUDASETDEVICEFLAGS-NEXT: The API is Removed.
// CUDASETDEVICEFLAGS-EMPTY:

/// Thread Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaThreadExit | FileCheck %s -check-prefix=CUDATHREADEXIT
// CUDATHREADEXIT: CUDA API:
// CUDATHREADEXIT-NEXT:   cudaThreadExit();
// CUDATHREADEXIT-NEXT: Is migrated to:
// CUDATHREADEXIT-NEXT:   dpct::get_current_device().reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaThreadSetLimit | FileCheck %s -check-prefix=CUDATHREADSETLIMIT
// CUDATHREADSETLIMIT: CUDA API:
// CUDATHREADSETLIMIT-NEXT:   cudaThreadSetLimit(l /*cudaLimit*/, s /*size_t*/);
// CUDATHREADSETLIMIT-NEXT: The API is Removed.
// CUDATHREADSETLIMIT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaThreadSynchronize | FileCheck %s -check-prefix=CUDATHREADSYNCHRONIZE
// CUDATHREADSYNCHRONIZE: CUDA API:
// CUDATHREADSYNCHRONIZE-NEXT:   cudaThreadSynchronize();
// CUDATHREADSYNCHRONIZE-NEXT: Is migrated to:
// CUDATHREADSYNCHRONIZE-NEXT:   dpct::get_current_device().queues_wait_and_throw();

/// Error Handling

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetLastError | FileCheck %s -check-prefix=CUDAGETLASTERROR
// CUDAGETLASTERROR: CUDA API:
// CUDAGETLASTERROR-NEXT:   cudaGetLastError();
// CUDAGETLASTERROR-NEXT: The API is Removed.
// CUDAGETLASTERROR-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaPeekAtLastError | FileCheck %s -check-prefix=CUDAPEEKATLASTERROR
// CUDAPEEKATLASTERROR: CUDA API:
// CUDAPEEKATLASTERROR-NEXT:   cudaPeekAtLastError();
// CUDAPEEKATLASTERROR-NEXT: The API is Removed.
// CUDAPEEKATLASTERROR-EMPTY:

/// Stream Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamAddCallback | FileCheck %s -check-prefix=CUDASTREAMADDCALLBACK
// CUDASTREAMADDCALLBACK: CUDA API:
// CUDASTREAMADDCALLBACK-NEXT:   cudaStream_t s;
// CUDASTREAMADDCALLBACK-NEXT:   cudaStreamAddCallback(s /*cudaStream_t*/, sc /*cudaStreamCallback_t*/,
// CUDASTREAMADDCALLBACK-NEXT:                         pData /*void **/, u /*unsigned int*/);
// CUDASTREAMADDCALLBACK-NEXT: Is migrated to:
// CUDASTREAMADDCALLBACK-NEXT:   dpct::queue_ptr s;
// CUDASTREAMADDCALLBACK-NEXT:   std::async([&]() { s->wait(); sc(s, 0, pData); });

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamAttachMemAsync | FileCheck %s -check-prefix=CUDASTREAMATTACHMEMASYNC
// CUDASTREAMATTACHMEMASYNC: CUDA API:
// CUDASTREAMATTACHMEMASYNC-NEXT:   cudaStreamAttachMemAsync(s /*cudaStream_t*/, pDev /*void **/, st /*size_t*/,
// CUDASTREAMATTACHMEMASYNC-NEXT:                            u /*unsigned int*/);
// CUDASTREAMATTACHMEMASYNC-NEXT: The API is Removed.
// CUDASTREAMATTACHMEMASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamBeginCapture | FileCheck %s -check-prefix=CUDASTREAMBEGINCAPTURE
// CUDASTREAMBEGINCAPTURE: CUDA API:
// CUDASTREAMBEGINCAPTURE-NEXT:   cudaStreamBeginCapture(s /*cudaStream_t*/, sc /*cudaStreamCaptureMode*/);
// CUDASTREAMBEGINCAPTURE-NEXT: The API is Removed.
// CUDASTREAMBEGINCAPTURE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamCreate | FileCheck %s -check-prefix=CUDASTREAMCREATE
// CUDASTREAMCREATE: CUDA API:
// CUDASTREAMCREATE-NEXT:   cudaStreamCreate(ps /*cudaStream_t **/);
// CUDASTREAMCREATE-NEXT: Is migrated to:
// CUDASTREAMCREATE-NEXT:   *(ps) = dpct::get_current_device().create_queue();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamCreateWithFlags | FileCheck %s -check-prefix=CUDASTREAMCREATEWITHFLAGS
// CUDASTREAMCREATEWITHFLAGS: CUDA API:
// CUDASTREAMCREATEWITHFLAGS-NEXT:   cudaStreamCreateWithFlags(ps /*cudaStream_t **/, u /*unsigned int*/);
// CUDASTREAMCREATEWITHFLAGS-NEXT: Is migrated to:
// CUDASTREAMCREATEWITHFLAGS-NEXT:   *(ps) = dpct::get_current_device().create_queue();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamCreateWithPriority | FileCheck %s -check-prefix=CUDASTREAMCREATEWITHPRIORITY
// CUDASTREAMCREATEWITHPRIORITY: CUDA API:
// CUDASTREAMCREATEWITHPRIORITY-NEXT:   cudaStreamCreateWithPriority(ps /*cudaStream_t **/, u /*unsigned int*/,
// CUDASTREAMCREATEWITHPRIORITY-NEXT:                                i /*int*/);
// CUDASTREAMCREATEWITHPRIORITY-NEXT: Is migrated to:
// CUDASTREAMCREATEWITHPRIORITY-NEXT:   *(ps) = dpct::get_current_device().create_queue();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamDestroy | FileCheck %s -check-prefix=CUDASTREAMDESTROY
// CUDASTREAMDESTROY: CUDA API:
// CUDASTREAMDESTROY-NEXT:   cudaStreamDestroy(s /*cudaStream_t*/);
// CUDASTREAMDESTROY-NEXT: Is migrated to:
// CUDASTREAMDESTROY-NEXT:   dpct::get_current_device().destroy_queue(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamEndCapture | FileCheck %s -check-prefix=CUDASTREAMENDCAPTURE
// CUDASTREAMENDCAPTURE: CUDA API:
// CUDASTREAMENDCAPTURE-NEXT:   cudaStreamEndCapture(s /*cudaStream_t*/, pg /*cudaGraph_t **/);
// CUDASTREAMENDCAPTURE-NEXT: The API is Removed.
// CUDASTREAMENDCAPTURE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamGetFlags | FileCheck %s -check-prefix=CUDASTREAMGETFLAGS
// CUDASTREAMGETFLAGS: CUDA API:
// CUDASTREAMGETFLAGS-NEXT:   cudaStreamGetFlags(s /*cudaStream_t*/, f /*unsigned int **/);
// CUDASTREAMGETFLAGS-NEXT: Is migrated to:
// CUDASTREAMGETFLAGS-NEXT:   *(f) = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamGetPriority | FileCheck %s -check-prefix=CUDASTREAMGETPRIORITY
// CUDASTREAMGETPRIORITY: CUDA API:
// CUDASTREAMGETPRIORITY-NEXT:   cudaStreamGetPriority(s /*cudaStream_t*/, pi /*int **/);
// CUDASTREAMGETPRIORITY-NEXT: Is migrated to:
// CUDASTREAMGETPRIORITY-NEXT:   *(pi) = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamIsCapturing | FileCheck %s -check-prefix=CUDASTREAMISCAPTURING
// CUDASTREAMISCAPTURING: CUDA API:
// CUDASTREAMISCAPTURING-NEXT:   cudaStreamIsCapturing(s /*cudaStream_t*/,
// CUDASTREAMISCAPTURING-NEXT:                         ps /* enum cudaStreamCaptureStatus **/);
// CUDASTREAMISCAPTURING-NEXT: The API is Removed.
// CUDASTREAMISCAPTURING-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamQuery | FileCheck %s -check-prefix=CUDASTREAMQUERY
// CUDASTREAMQUERY: CUDA API:
// CUDASTREAMQUERY-NEXT:   cudaStreamQuery(s /*cudaStream_t*/);
// CUDASTREAMQUERY-NEXT: The API is Removed.
// CUDASTREAMQUERY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamSynchronize | FileCheck %s -check-prefix=CUDASTREAMSYNCHRONIZE
// CUDASTREAMSYNCHRONIZE: CUDA API:
// CUDASTREAMSYNCHRONIZE-NEXT:   cudaStream_t s;
// CUDASTREAMSYNCHRONIZE-NEXT:   cudaStreamSynchronize(s /*cudaStream_t*/);
// CUDASTREAMSYNCHRONIZE-NEXT: Is migrated to:
// CUDASTREAMSYNCHRONIZE-NEXT:   dpct::queue_ptr s;
// CUDASTREAMSYNCHRONIZE-NEXT:   s->wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaStreamWaitEvent | FileCheck %s -check-prefix=CUDASTREAMWAITEVENT
// CUDASTREAMWAITEVENT: CUDA API:
// CUDASTREAMWAITEVENT-NEXT:   cudaStream_t s;
// CUDASTREAMWAITEVENT-NEXT:   cudaStreamWaitEvent(s /*cudaStream_t*/, e /*cudaEvent_t*/,
// CUDASTREAMWAITEVENT-NEXT:                       u /*unsigned int*/);
// CUDASTREAMWAITEVENT-NEXT: Is migrated to:
// CUDASTREAMWAITEVENT-NEXT:   dpct::queue_ptr s;
// CUDASTREAMWAITEVENT-NEXT:   s->ext_oneapi_submit_barrier({*e});

/// Event Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventCreate | FileCheck %s -check-prefix=CUDAEVENTCREATE
// CUDAEVENTCREATE: CUDA API:
// CUDAEVENTCREATE-NEXT:   cudaEventCreate(pe /*cudaEvent_t **/);
// CUDAEVENTCREATE-NEXT: Is migrated to:
// CUDAEVENTCREATE-NEXT:   *pe = new sycl::event();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventCreateWithFlags | FileCheck %s -check-prefix=CUDAEVENTCREATEWITHFLAGS
// CUDAEVENTCREATEWITHFLAGS: CUDA API:
// CUDAEVENTCREATEWITHFLAGS-NEXT:   cudaEventCreateWithFlags(pe /*cudaEvent_t **/, u /*unsigned int*/);
// CUDAEVENTCREATEWITHFLAGS-NEXT: Is migrated to:
// CUDAEVENTCREATEWITHFLAGS-NEXT:   *pe = new sycl::event();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventDestroy | FileCheck %s -check-prefix=CUDAEVENTDESTROY
// CUDAEVENTDESTROY: CUDA API:
// CUDAEVENTDESTROY-NEXT:   cudaEventDestroy(e /*cudaEvent_t*/);
// CUDAEVENTDESTROY-NEXT: Is migrated to:
// CUDAEVENTDESTROY-NEXT:   dpct::destroy_event(e);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventElapsedTime | FileCheck %s -check-prefix=CUDAEVENTELAPSEDTIME
// CUDAEVENTELAPSEDTIME: CUDA API:
// CUDAEVENTELAPSEDTIME-NEXT:   cudaEventElapsedTime(pf /*float **/, e1 /*cudaEvent_t*/, e2 /*cudaEvent_t*/);
// CUDAEVENTELAPSEDTIME-NEXT: Is migrated to:
// CUDAEVENTELAPSEDTIME-NEXT:   *(pf) = std::chrono::duration<float, std::milli>(e2_ct1 - e1_ct1).count();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventQuery | FileCheck %s -check-prefix=CUDAEVENTQUERY
// CUDAEVENTQUERY: CUDA API:
// CUDAEVENTQUERY-NEXT:   cudaEvent_t e;
// CUDAEVENTQUERY-NEXT:   cudaEventQuery(e /*cudaEvent_t*/);
// CUDAEVENTQUERY-NEXT: Is migrated to:
// CUDAEVENTQUERY-NEXT:   dpct::event_ptr e;
// CUDAEVENTQUERY-NEXT:   (int)e->get_info<sycl::info::event::command_execution_status>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventRecord | FileCheck %s -check-prefix=CUDAEVENTRECORD
// CUDAEVENTRECORD: CUDA API:
// CUDAEVENTRECORD-NEXT:   cudaStream_t s;
// CUDAEVENTRECORD-NEXT:   cudaEventRecord(e /*cudaEvent_t*/, s /*cudaStream_t*/);
// CUDAEVENTRECORD-NEXT: Is migrated to:
// CUDAEVENTRECORD-NEXT:   dpct::queue_ptr s;
// CUDAEVENTRECORD-NEXT:   *e = s->ext_oneapi_submit_barrier();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaEventSynchronize | FileCheck %s -check-prefix=CUDAEVENTSYNCHRONIZE
// CUDAEVENTSYNCHRONIZE: CUDA API:
// CUDAEVENTSYNCHRONIZE-NEXT:   cudaEvent_t e;
// CUDAEVENTSYNCHRONIZE-NEXT:   cudaEventSynchronize(e /*cudaEvent_t*/);
// CUDAEVENTSYNCHRONIZE-NEXT: Is migrated to:
// CUDAEVENTSYNCHRONIZE-NEXT:   dpct::event_ptr e;
// CUDAEVENTSYNCHRONIZE-NEXT:   e->wait_and_throw();

/// Execution Control

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFuncGetAttributes | FileCheck %s -check-prefix=CUDAFUNCGETATTRIBUTES
// CUDAFUNCGETATTRIBUTES: CUDA API:
// CUDAFUNCGETATTRIBUTES-NEXT:   cudaFuncAttributes *attr;
// CUDAFUNCGETATTRIBUTES-NEXT:   cudaFuncGetAttributes(attr, f /*const void **/);
// CUDAFUNCGETATTRIBUTES-NEXT: Is migrated to:
// CUDAFUNCGETATTRIBUTES-NEXT:   dpct::kernel_function_info *attr;
// CUDAFUNCGETATTRIBUTES-NEXT:   dpct::get_kernel_function_info(attr, (const void *)f /*const void **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFuncSetCacheConfig | FileCheck %s -check-prefix=CUDAFUNCSETCACHECONFIG
// CUDAFUNCSETCACHECONFIG: CUDA API:
// CUDAFUNCSETCACHECONFIG-NEXT:   cudaFuncSetCacheConfig(pFunc /*const void **/, f /*cudaFuncCache*/);
// CUDAFUNCSETCACHECONFIG-NEXT: The API is Removed.
// CUDAFUNCSETCACHECONFIG-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFuncSetSharedMemConfig | FileCheck %s -check-prefix=CUDAFUNCSETSHAREDMEMCONFIG
// CUDAFUNCSETSHAREDMEMCONFIG: CUDA API:
// CUDAFUNCSETSHAREDMEMCONFIG-NEXT:   cudaFuncSetSharedMemConfig(pFunc /*const void **/, s /*cudaSharedMemConfig*/);
// CUDAFUNCSETSHAREDMEMCONFIG-NEXT: The API is Removed.
// CUDAFUNCSETSHAREDMEMCONFIG-EMPTY:

/// Occupancy

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaOccupancyMaxActiveBlocksPerMultiprocessor | FileCheck %s -check-prefix=CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR
// CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR: CUDA API:
// CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR-NEXT:   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
// CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR-NEXT:       pi /*int **/, pFunc /*const void **/, i /*int*/, s /*size_t*/);
// CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR-NEXT: Is migrated to (with the option --use-experimental-features=occupancy-calculation):
// CUDAOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR-NEXT:   dpct::experimental::calculate_max_active_wg_per_xecore(pi, i, s + dpct_placeholder /* total share local memory size */);

/// Memory Management [DEPRECATED]

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyArrayToArray | FileCheck %s -check-prefix=CUDAMEMCPYARRAYTOARRAY
// CUDAMEMCPYARRAYTOARRAY: CUDA API:
// CUDAMEMCPYARRAYTOARRAY-NEXT:   cudaArray_t dst;
// CUDAMEMCPYARRAYTOARRAY-NEXT:   cudaArray_t src;
// CUDAMEMCPYARRAYTOARRAY-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPYARRAYTOARRAY-NEXT:   cudaMemcpyArrayToArray(dst, s1 /*size_t*/, s2 /*size_t*/, src, s3 /*size_t*/,
// CUDAMEMCPYARRAYTOARRAY-NEXT:                          s4 /*size_t*/, s5 /*size_t*/, m);
// CUDAMEMCPYARRAYTOARRAY-NEXT: Is migrated to:
// CUDAMEMCPYARRAYTOARRAY-NEXT:   dpct::image_matrix_p dst;
// CUDAMEMCPYARRAYTOARRAY-NEXT:   dpct::image_matrix_p src;
// CUDAMEMCPYARRAYTOARRAY-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPYARRAYTOARRAY-NEXT:   dpct::dpct_memcpy(dst->to_pitched_data(), sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/, src->to_pitched_data(), sycl::id<3>(s3 /*size_t*/,
// CUDAMEMCPYARRAYTOARRAY-NEXT:                          s4, 0) /*size_t*/, sycl::range<3>(s5, 1, 1));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyFromArray | FileCheck %s -check-prefix=CUDAMEMCPYFROMARRAY
// CUDAMEMCPYFROMARRAY: CUDA API:
// CUDAMEMCPYFROMARRAY-NEXT:   cudaArray_t src;
// CUDAMEMCPYFROMARRAY-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPYFROMARRAY-NEXT:   cudaMemcpyFromArray(dst /*void **/, src, s1 /*size_t*/, s2 /*size_t*/,
// CUDAMEMCPYFROMARRAY-NEXT:                       s3 /*size_t*/, m);
// CUDAMEMCPYFROMARRAY-NEXT: Is migrated to:
// CUDAMEMCPYFROMARRAY-NEXT:   dpct::image_matrix_p src;
// CUDAMEMCPYFROMARRAY-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPYFROMARRAY-NEXT:   dpct::dpct_memcpy(dpct::pitched_data(dst, s3, s3, 1) /*void **/, sycl::id<3>(0, 0, 0), src->to_pitched_data(), sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/,
// CUDAMEMCPYFROMARRAY-NEXT:                       sycl::range<3>(s3, 1, 1));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyFromArrayAsync | FileCheck %s -check-prefix=CUDAMEMCPYFROMARRAYASYNC
// CUDAMEMCPYFROMARRAYASYNC: CUDA API:
// CUDAMEMCPYFROMARRAYASYNC-NEXT:   cudaArray_t src;
// CUDAMEMCPYFROMARRAYASYNC-NEXT:   cudaStream_t s;
// CUDAMEMCPYFROMARRAYASYNC-NEXT:   cudaMemcpyFromArrayAsync(dst /*void **/, src, s1 /*size_t*/, s2 /*size_t*/,
// CUDAMEMCPYFROMARRAYASYNC-NEXT:                            s3 /*size_t*/, m /*cudaMemcpyKind*/, s);
// CUDAMEMCPYFROMARRAYASYNC-NEXT: Is migrated to:
// CUDAMEMCPYFROMARRAYASYNC-NEXT:   dpct::image_matrix_p src;
// CUDAMEMCPYFROMARRAYASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMCPYFROMARRAYASYNC-NEXT:   dpct::async_dpct_memcpy(dpct::pitched_data(dst, s3, s3, 1) /*void **/, sycl::id<3>(0, 0, 0), src->to_pitched_data(), sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/,
// CUDAMEMCPYFROMARRAYASYNC-NEXT:                            sycl::range<3>(s3, 1, 1), dpct::automatic, *s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyToArray | FileCheck %s -check-prefix=CUDAMEMCPYTOARRAY
// CUDAMEMCPYTOARRAY: CUDA API:
// CUDAMEMCPYTOARRAY-NEXT:   cudaArray_t dst;
// CUDAMEMCPYTOARRAY-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPYTOARRAY-NEXT:   cudaMemcpyToArray(dst, s1 /*size_t*/, s2 /*size_t*/, src /*const void **/,
// CUDAMEMCPYTOARRAY-NEXT:                     s3 /*size_t*/, m);
// CUDAMEMCPYTOARRAY-NEXT: Is migrated to:
// CUDAMEMCPYTOARRAY-NEXT:   dpct::image_matrix_p dst;
// CUDAMEMCPYTOARRAY-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPYTOARRAY-NEXT:   dpct::dpct_memcpy(dst->to_pitched_data(), sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/, dpct::pitched_data(src, s3, s3, 1) /*const void **/,
// CUDAMEMCPYTOARRAY-NEXT:                     sycl::id<3>(0, 0, 0), sycl::range<3>(s3, 1, 1));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyToArrayAsync | FileCheck %s -check-prefix=CUDAMEMCPYTOARRAYASYNC
// CUDAMEMCPYTOARRAYASYNC: CUDA API:
// CUDAMEMCPYTOARRAYASYNC:   cudaArray_t dst;
// CUDAMEMCPYTOARRAYASYNC:   cudaStream_t s;
// CUDAMEMCPYTOARRAYASYNC:   cudaMemcpyToArrayAsync(dst, s1 /*size_t*/, s2 /*size_t*/, src /*const void **/,
// CUDAMEMCPYTOARRAYASYNC:                          s3 /*size_t*/, m /*cudaMemcpyKind*/, s);
// CUDAMEMCPYTOARRAYASYNC-NEXT: Is migrated to:
// CUDAMEMCPYTOARRAYASYNC:   dpct::image_matrix_p dst;
// CUDAMEMCPYTOARRAYASYNC:   dpct::queue_ptr s;
// CUDAMEMCPYTOARRAYASYNC:   dpct::async_dpct_memcpy(dst->to_pitched_data(), sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/, dpct::pitched_data(src, s3, s3, 1) /*const void **/,
// CUDAMEMCPYTOARRAYASYNC:                          sycl::id<3>(0, 0, 0), sycl::range<3>(s3, 1, 1), dpct::automatic, *s);

/// Unified Addressing

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaPointerGetAttributes | FileCheck %s -check-prefix=CUDAPOINTERGETATTRIBUTES
// CUDAPOINTERGETATTRIBUTES: CUDA API:
// CUDAPOINTERGETATTRIBUTES-NEXT:   const void *ptr;
// CUDAPOINTERGETATTRIBUTES-NEXT:   cudaPointerGetAttributes(attr /*cudaPointerAttributes **/,
// CUDAPOINTERGETATTRIBUTES-NEXT:                            ptr /*const void **/);
// CUDAPOINTERGETATTRIBUTES-NEXT: Is migrated to:
// CUDAPOINTERGETATTRIBUTES-NEXT:   const void *ptr;
// CUDAPOINTERGETATTRIBUTES-NEXT:   attr->init(ptr);

/// Peer Device Memory Access

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceCanAccessPeer | FileCheck %s -check-prefix=CUDADEVICECANACCESSPEER
// CUDADEVICECANACCESSPEER: CUDA API:
// CUDADEVICECANACCESSPEER-NEXT:   cudaDeviceCanAccessPeer(pi /*int **/, i1 /*int*/, i2 /*int*/);
// CUDADEVICECANACCESSPEER-NEXT: Is migrated to:
// CUDADEVICECANACCESSPEER-NEXT:   *pi = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceDisablePeerAccess | FileCheck %s -check-prefix=CUDADEVICEDISABLEPEERACCESS
// CUDADEVICEDISABLEPEERACCESS: CUDA API:
// CUDADEVICEDISABLEPEERACCESS-NEXT:   cudaDeviceDisablePeerAccess(i /*int*/);
// CUDADEVICEDISABLEPEERACCESS-NEXT: The API is Removed.
// CUDADEVICEDISABLEPEERACCESS-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDeviceEnablePeerAccess | FileCheck %s -check-prefix=CUDADEVICEENABLEPEERACCESS
// CUDADEVICEENABLEPEERACCESS: CUDA API:
// CUDADEVICEENABLEPEERACCESS-NEXT:   cudaDeviceEnablePeerAccess(i /*int*/, u /*unsigned int*/);
// CUDADEVICEENABLEPEERACCESS-NEXT: The API is Removed.
// CUDADEVICEENABLEPEERACCESS-EMPTY:

/// Texture Object Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaCreateChannelDesc | FileCheck %s -check-prefix=CUDACREATECHANNELDESC
// CUDACREATECHANNELDESC: CUDA API:
// CUDACREATECHANNELDESC-NEXT:   cudaCreateChannelDesc(i1 /*int*/, i2 /*int*/, i3 /*int*/, i4 /*int*/,
// CUDACREATECHANNELDESC-NEXT:                         c /*cudaChannelFormatKind*/);
// CUDACREATECHANNELDESC-NEXT: Is migrated to:
// CUDACREATECHANNELDESC-NEXT:   dpct::image_channel(i1, i2, i3, i4, c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaCreateTextureObject | FileCheck %s -check-prefix=CUDACREATETEXTUREOBJECT
// CUDACREATETEXTUREOBJECT: CUDA API:
// CUDACREATETEXTUREOBJECT-NEXT:   cudaCreateTextureObject(
// CUDACREATETEXTUREOBJECT-NEXT:       pto /*cudaTextureObject_t **/, prd /*const cudaResourceDesc **/,
// CUDACREATETEXTUREOBJECT-NEXT:       ptd /*const cudaTextureDesc **/, prvd /*const cudaResourceViewDesc **/);
// CUDACREATETEXTUREOBJECT-NEXT: Is migrated to:
// CUDACREATETEXTUREOBJECT-NEXT:   *pto = dpct::create_image_wrapper(*prd, *ptd);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDestroyTextureObject | FileCheck %s -check-prefix=CUDADESTROYTEXTUREOBJECT
// CUDADESTROYTEXTUREOBJECT: CUDA API:
// CUDADESTROYTEXTUREOBJECT-NEXT:   cudaDestroyTextureObject(t /*cudaTextureObject_t*/);
// CUDADESTROYTEXTUREOBJECT-NEXT: Is migrated to:
// CUDADESTROYTEXTUREOBJECT-NEXT:   delete t;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetChannelDesc | FileCheck %s -check-prefix=CUDAGETCHANNELDESC
// CUDAGETCHANNELDESC: CUDA API:
// CUDAGETCHANNELDESC-NEXT:   cudaArray_t a;
// CUDAGETCHANNELDESC-NEXT:   cudaGetChannelDesc(pc /*cudaChannelFormatDesc **/, a);
// CUDAGETCHANNELDESC-NEXT: Is migrated to:
// CUDAGETCHANNELDESC-NEXT:   dpct::image_matrix_p a;
// CUDAGETCHANNELDESC-NEXT:   *pc = a->get_channel();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetTextureObjectResourceDesc | FileCheck %s -check-prefix=CUDAGETTEXTUREOBJECTRESOURCEDESC
// CUDAGETTEXTUREOBJECTRESOURCEDESC: CUDA API:
// CUDAGETTEXTUREOBJECTRESOURCEDESC-NEXT:   cudaTextureObject_t t;
// CUDAGETTEXTUREOBJECTRESOURCEDESC-NEXT:   cudaGetTextureObjectResourceDesc(pr /*cudaResourceDesc **/,
// CUDAGETTEXTUREOBJECTRESOURCEDESC-NEXT:                                    t /*cudaTextureObject_t*/);
// CUDAGETTEXTUREOBJECTRESOURCEDESC-NEXT: Is migrated to:
// CUDAGETTEXTUREOBJECTRESOURCEDESC-NEXT:   dpct::image_wrapper_base_p t;
// CUDAGETTEXTUREOBJECTRESOURCEDESC-NEXT:   *pr = t->get_data();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetTextureObjectTextureDesc | FileCheck %s -check-prefix=CUDAGETTEXTUREOBJECTTEXTUREDESC
// CUDAGETTEXTUREOBJECTTEXTUREDESC: CUDA API:
// CUDAGETTEXTUREOBJECTTEXTUREDESC-NEXT:   cudaTextureObject_t t;
// CUDAGETTEXTUREOBJECTTEXTUREDESC-NEXT:   cudaGetTextureObjectTextureDesc(pt /*cudaTextureDesc **/,
// CUDAGETTEXTUREOBJECTTEXTUREDESC-NEXT:                                   t /*cudaTextureObject_t*/);
// CUDAGETTEXTUREOBJECTTEXTUREDESC-NEXT: Is migrated to:
// CUDAGETTEXTUREOBJECTTEXTUREDESC-NEXT:   dpct::image_wrapper_base_p t;
// CUDAGETTEXTUREOBJECTTEXTUREDESC-NEXT:   *pt = t->get_sampling_info();

/// Version Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaDriverGetVersion | FileCheck %s -check-prefix=CUDADRIVERGETVERSION
// CUDADRIVERGETVERSION: CUDA API:
// CUDADRIVERGETVERSION-NEXT:   cudaDriverGetVersion(pi /*int **/);
// CUDADRIVERGETVERSION-NEXT: Is migrated to:
// CUDADRIVERGETVERSION-NEXT:   *pi = dpct::get_major_version(dpct::get_current_device());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaRuntimeGetVersion | FileCheck %s -check-prefix=CUDARUNTIMEGETVERSION
// CUDARUNTIMEGETVERSION: CUDA API:
// CUDARUNTIMEGETVERSION-NEXT:   cudaRuntimeGetVersion(pi /*int **/);
// CUDARUNTIMEGETVERSION-NEXT: Is migrated to:
// CUDARUNTIMEGETVERSION-NEXT:   *pi = dpct::get_major_version(dpct::get_current_device());
