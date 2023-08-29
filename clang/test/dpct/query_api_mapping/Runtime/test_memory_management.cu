/// Memory Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaArrayGetInfo | FileCheck %s -check-prefix=CUDAARRAYGETINFO
// CUDAARRAYGETINFO: CUDA API:
// CUDAARRAYGETINFO-NEXT:   cudaArray_t a;
// CUDAARRAYGETINFO-NEXT:   cudaArrayGetInfo(c /*cudaChannelFormatDesc **/, e /*cudaExtent **/,
// CUDAARRAYGETINFO-NEXT:                    u /*unsigned int **/, a /*cudaArray_t*/);
// CUDAARRAYGETINFO-NEXT: Is migrated to:
// CUDAARRAYGETINFO-NEXT:   dpct::image_matrix_p a;
// CUDAARRAYGETINFO-NEXT:   *c = a->get_channel();
// CUDAARRAYGETINFO-NEXT:   *e = a->get_range();
// CUDAARRAYGETINFO-NEXT:   *u = 0;
// CUDAARRAYGETINFO-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFree | FileCheck %s -check-prefix=CUDAFREE
// CUDAFREE: CUDA API:
// CUDAFREE-NEXT:   cudaFree(pv /*void **/);
// CUDAFREE-NEXT: Is migrated to:
// CUDAFREE-NEXT:   sycl::free(pv, dpct::get_default_queue());
// CUDAFREE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFreeArray | FileCheck %s -check-prefix=CUDAFREEARRAY
// CUDAFREEARRAY: CUDA API:
// CUDAFREEARRAY-NEXT:   cudaFreeArray(a /*cudaArray_t*/);
// CUDAFREEARRAY-NEXT: Is migrated to:
// CUDAFREEARRAY-NEXT:   delete a;
// CUDAFREEARRAY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFreeHost | FileCheck %s -check-prefix=CUDAFREEHOST
// CUDAFREEHOST: CUDA API:
// CUDAFREEHOST-NEXT:   cudaFreeHost(pv /*void **/);
// CUDAFREEHOST-NEXT: Is migrated to:
// CUDAFREEHOST-NEXT:   sycl::free(pv, dpct::get_default_queue());
// CUDAFREEHOST-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetSymbolAddress | FileCheck %s -check-prefix=CUDAGETSYMBOLADDRESS
// CUDAGETSYMBOLADDRESS: CUDA API:
// CUDAGETSYMBOLADDRESS-NEXT:   const void *pv;
// CUDAGETSYMBOLADDRESS-NEXT:   cudaGetSymbolAddress(ppv /*void ***/, pv /*const void **/);
// CUDAGETSYMBOLADDRESS-NEXT: Is migrated to:
// CUDAGETSYMBOLADDRESS-NEXT:   const void *pv;
// CUDAGETSYMBOLADDRESS-NEXT:   *(ppv) = pv.get_ptr();
// CUDAGETSYMBOLADDRESS-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetSymbolSize | FileCheck %s -check-prefix=CUDAGETSYMBOLSIZE
// CUDAGETSYMBOLSIZE: CUDA API:
// CUDAGETSYMBOLSIZE-NEXT:   const void *pv;
// CUDAGETSYMBOLSIZE-NEXT:   cudaGetSymbolSize(s /*size_t **/, pv /*const void **/);
// CUDAGETSYMBOLSIZE-NEXT: Is migrated to:
// CUDAGETSYMBOLSIZE-NEXT:   const void *pv;
// CUDAGETSYMBOLSIZE-NEXT:   *s = pv.get_size();
// CUDAGETSYMBOLSIZE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostAlloc | FileCheck %s -check-prefix=CUDAHOSTALLOC
// CUDAHOSTALLOC: CUDA API:
// CUDAHOSTALLOC-NEXT:   cudaHostAlloc(ppv /*void ***/, s /*size_t*/, u /*unsigned int*/);
// CUDAHOSTALLOC-NEXT: Is migrated to:
// CUDAHOSTALLOC-NEXT:   *ppv = (void *)sycl::malloc_host(s, dpct::get_default_queue());
// CUDAHOSTALLOC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostGetDevicePointer | FileCheck %s -check-prefix=CUDAHOSTGETDEVICEPOINTER
// CUDAHOSTGETDEVICEPOINTER: CUDA API:
// CUDAHOSTGETDEVICEPOINTER-NEXT:   cudaHostGetDevicePointer(ppv /*void ***/, pv /*void **/, u /*unsigned int*/);
// CUDAHOSTGETDEVICEPOINTER-NEXT: Is migrated to:
// CUDAHOSTGETDEVICEPOINTER-NEXT:   *ppv = (void *)pv;
// CUDAHOSTGETDEVICEPOINTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostGetFlags | FileCheck %s -check-prefix=CUDAHOSTGETFLAGS
// CUDAHOSTGETFLAGS: CUDA API:
// CUDAHOSTGETFLAGS-NEXT:   cudaHostGetFlags(pu /*unsigned int **/, pv /*void **/);
// CUDAHOSTGETFLAGS-NEXT: Is migrated to:
// CUDAHOSTGETFLAGS-NEXT:   *pu = 0;
// CUDAHOSTGETFLAGS-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostRegister | FileCheck %s -check-prefix=CUDAHOSTREGISTER
// CUDAHOSTREGISTER: CUDA API:
// CUDAHOSTREGISTER-NEXT:   cudaHostRegister(pv /*void **/, s /*size_t*/, u /*unsigned int*/);
// CUDAHOSTREGISTER-NEXT: The API is Removed.
// CUDAHOSTREGISTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostUnregister | FileCheck %s -check-prefix=CUDAHOSTUNREGISTER
// CUDAHOSTUNREGISTER: CUDA API:
// CUDAHOSTUNREGISTER-NEXT:   cudaHostUnregister(pv /*void **/);
// CUDAHOSTUNREGISTER-NEXT: The API is Removed.
// CUDAHOSTUNREGISTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMalloc | FileCheck %s -check-prefix=CUDAMALLOC
// CUDAMALLOC: CUDA API:
// CUDAMALLOC-NEXT:   cudaMalloc(ppv /*void ***/, s /*size_t*/);
// CUDAMALLOC-NEXT: Is migrated to:
// CUDAMALLOC-NEXT:   *ppv = (void *)sycl::malloc_device(s, dpct::get_default_queue());
// CUDAMALLOC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMalloc3D | FileCheck %s -check-prefix=CUDAMALLOC3D
// CUDAMALLOC3D: CUDA API:
// CUDAMALLOC3D-NEXT:   cudaMalloc3D(pp /*cudaPitchedPtr **/, e /*cudaExtent*/);
// CUDAMALLOC3D-NEXT: Is migrated to:
// CUDAMALLOC3D-NEXT:   *pp = dpct::dpct_malloc(e /*cudaExtent*/);
// CUDAMALLOC3D-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMalloc3DArray | FileCheck %s -check-prefix=CUDAMALLOC3DARRAY
// CUDAMALLOC3DARRAY: CUDA API:
// CUDAMALLOC3DARRAY-NEXT:   cudaMalloc3DArray(pa /*cudaArray_t **/, pc /*cudaChannelFormatDesc **/,
// CUDAMALLOC3DARRAY-NEXT:                     e /*cudaExtent*/, u /*unsigned int*/);
// CUDAMALLOC3DARRAY-NEXT: Is migrated to:
// CUDAMALLOC3DARRAY-NEXT:   *pa = new dpct::image_matrix(*pc /*cudaChannelFormatDesc **/,
// CUDAMALLOC3DARRAY-NEXT:                                e /*unsigned int*/);
// CUDAMALLOC3DARRAY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMallocArray | FileCheck %s -check-prefix=CUDAMALLOCARRAY
// CUDAMALLOCARRAY: CUDA API:
// CUDAMALLOCARRAY-NEXT:   cudaMallocArray(pa /*cudaArray_t **/, pc /*cudaChannelFormatDesc **/,
// CUDAMALLOCARRAY-NEXT:                   s1 /*size_t*/, s2 /*size_t*/, u /*unsigned int*/);
// CUDAMALLOCARRAY-NEXT: Is migrated to:
// CUDAMALLOCARRAY-NEXT:   *pa = new dpct::image_matrix(*pc /*cudaChannelFormatDesc **/,
// CUDAMALLOCARRAY-NEXT:                                sycl::range<2>(s1 /*size_t*/, s2) /*unsigned int*/);
// CUDAMALLOCARRAY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMallocHost | FileCheck %s -check-prefix=CUDAMALLOCHOST
// CUDAMALLOCHOST: CUDA API:
// CUDAMALLOCHOST-NEXT:   cudaMallocHost(ppv /*void ***/, s /*size_t*/);
// CUDAMALLOCHOST-NEXT: Is migrated to:
// CUDAMALLOCHOST-NEXT:   *ppv = (void *)sycl::malloc_host(s, dpct::get_default_queue());
// CUDAMALLOCHOST-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMallocManaged | FileCheck %s -check-prefix=CUDAMALLOCMANAGED
// CUDAMALLOCMANAGED: CUDA API:
// CUDAMALLOCMANAGED-NEXT:   cudaMallocManaged(ppv /*void ***/, s /*size_t*/, u /*unsigned int*/);
// CUDAMALLOCMANAGED-NEXT: Is migrated to:
// CUDAMALLOCMANAGED-NEXT:   *ppv = (void *)sycl::malloc_shared(s, dpct::get_default_queue());
// CUDAMALLOCMANAGED-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMallocPitch | FileCheck %s -check-prefix=CUDAMALLOCPITCH
// CUDAMALLOCPITCH: CUDA API:
// CUDAMALLOCPITCH-NEXT:   cudaMallocPitch(ppv /*void ***/, pz /*size_t **/, s1 /*size_t*/,
// CUDAMALLOCPITCH-NEXT:                   s2 /*size_t*/);
// CUDAMALLOCPITCH-NEXT: Is migrated to:
// CUDAMALLOCPITCH-NEXT:   *ppv = dpct::dpct_malloc(*pz, s1, s2);
// CUDAMALLOCPITCH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemAdvise | FileCheck %s -check-prefix=CUDAMEMADVISE
// CUDAMEMADVISE: CUDA API:
// CUDAMEMADVISE-NEXT:   cudaMemAdvise(pv /*const void **/, s /*size_t*/, m /*cudaMemoryAdvise*/,
// CUDAMEMADVISE-NEXT:                 i /*int*/);
// CUDAMEMADVISE-NEXT: Is migrated to:
// CUDAMEMADVISE-NEXT:   dpct::get_device(i).default_queue().mem_advise(pv, s, m);
// CUDAMEMADVISE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemGetInfo | FileCheck %s -check-prefix=CUDAMEMGETINFO
// CUDAMEMGETINFO: CUDA API:
// CUDAMEMGETINFO-NEXT:   cudaMemGetInfo(ps1 /*size_t **/, ps2 /*size_t **/);
// CUDAMEMGETINFO-NEXT: Is migrated to:
// CUDAMEMGETINFO-NEXT:   dpct::get_current_device().get_memory_info(*ps1, *ps2);
// CUDAMEMGETINFO-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemPrefetchAsync | FileCheck %s -check-prefix=CUDAMEMPREFETCHASYNC
// CUDAMEMPREFETCHASYNC: CUDA API:
// CUDAMEMPREFETCHASYNC-NEXT:   cudaStream_t cs;
// CUDAMEMPREFETCHASYNC-NEXT:   cudaMemPrefetchAsync(pv /*const void **/, s /*size_t*/, i /*int*/,
// CUDAMEMPREFETCHASYNC-NEXT:                        cs /*cudaStream_t*/);
// CUDAMEMPREFETCHASYNC-NEXT: Is migrated to:
// CUDAMEMPREFETCHASYNC-NEXT:   dpct::queue_ptr cs;
// CUDAMEMPREFETCHASYNC-NEXT:   cs->prefetch(pv,s);
// CUDAMEMPREFETCHASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy | FileCheck %s -check-prefix=CUDAMEMCPY
// CUDAMEMCPY: CUDA API:
// CUDAMEMCPY-NEXT:   cudaMemcpy(pv /*void **/, cpv /*const void **/, s /*size_t*/,
// CUDAMEMCPY-NEXT:              m /*cudaMemcpyKind*/);
// CUDAMEMCPY-NEXT: Is migrated to:
// CUDAMEMCPY-NEXT:   dpct::get_default_queue().memcpy(pv /*void **/, cpv /*const void **/, s /*cudaMemcpyKind*/).wait();
// CUDAMEMCPY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2D | FileCheck %s -check-prefix=CUDAMEMCPY2D
// CUDAMEMCPY2D: CUDA API:
// CUDAMEMCPY2D-NEXT:   cudaMemcpy2D(pv /*void **/, s1 /*size_t*/, cpv /*const void **/,
// CUDAMEMCPY2D-NEXT:                s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/,
// CUDAMEMCPY2D-NEXT:                m /*cudaMemcpyKind*/);
// CUDAMEMCPY2D-NEXT: Is migrated to:
// CUDAMEMCPY2D-NEXT:   dpct::dpct_memcpy(pv /*void **/, s1 /*size_t*/, cpv /*const void **/,
// CUDAMEMCPY2D-NEXT:                     s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/,
// CUDAMEMCPY2D-NEXT:                     m /*cudaMemcpyKind*/);
// CUDAMEMCPY2D-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DArrayToArray | FileCheck %s -check-prefix=CUDAMEMCPY2DARRAYTOARRAY
// CUDAMEMCPY2DARRAYTOARRAY: CUDA API:
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   cudaMemcpy2DArrayToArray(a /*cudaArray_t*/, s1 /*size_t*/, s2 /*size_t*/,
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:                            ac /*cudaArray_const_t*/, s3 /*size_t*/,
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:                            s4 /*size_t*/, s5 /*size_t*/, s6 /*size_t*/,
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:                            m /*cudaMemcpyKind*/);
// CUDAMEMCPY2DARRAYTOARRAY-NEXT: Is migrated to:
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   dpct::dpct_memcpy(a->to_pitched_data() /*cudaArray_t*/, sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/,
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:                     ac->to_pitched_data() /*cudaArray_const_t*/, sycl::id<3>(s3 /*size_t*/,
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:                     s4, 0) /*size_t*/, sycl::range<3>(s5 /*size_t*/, s6, 1) /*cudaMemcpyKind*/);
// CUDAMEMCPY2DARRAYTOARRAY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DAsync | FileCheck %s -check-prefix=CUDAMEMCPY2DASYNC
// CUDAMEMCPY2DASYNC: CUDA API:
// CUDAMEMCPY2DASYNC-NEXT:   cudaMemcpy2DAsync(pv /*void **/, s1 /*size_t*/, cpv /*const void **/,
// CUDAMEMCPY2DASYNC-NEXT:                     s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/,
// CUDAMEMCPY2DASYNC-NEXT:                     m /*cudaMemoryAdvise*/, s /*cudaStream_t*/);
// CUDAMEMCPY2DASYNC-NEXT: Is migrated to:
// CUDAMEMCPY2DASYNC-NEXT:   dpct::async_dpct_memcpy(pv /*void **/, s1 /*size_t*/, cpv /*const void **/,
// CUDAMEMCPY2DASYNC-NEXT:                           s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/,
// CUDAMEMCPY2DASYNC-NEXT:                           m /*cudaMemoryAdvise*/, *s /*cudaStream_t*/);
// CUDAMEMCPY2DASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DFromArray | FileCheck %s -check-prefix=CUDAMEMCPY2DFROMARRAY
// CUDAMEMCPY2DFROMARRAY: CUDA API:
// CUDAMEMCPY2DFROMARRAY-NEXT:   cudaMemcpy2DFromArray(pv /*void **/, s1 /*size_t*/, a /*cudaArray_const_t*/,
// CUDAMEMCPY2DFROMARRAY-NEXT:                         s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/,
// CUDAMEMCPY2DFROMARRAY-NEXT:                         s5 /*size_t*/, m /*cudaMemcpyKind*/);
// CUDAMEMCPY2DFROMARRAY-NEXT: Is migrated to:
// CUDAMEMCPY2DFROMARRAY-NEXT:   dpct::dpct_memcpy(dpct::pitched_data(pv /*void **/, s1, s1, 1) /*size_t*/, sycl::id<3>(0, 0, 0), a->to_pitched_data() /*cudaArray_const_t*/,
// CUDAMEMCPY2DFROMARRAY-NEXT:                     sycl::id<3>(s2 /*size_t*/, s3, 0) /*size_t*/, sycl::range<3>(s4 /*size_t*/,
// CUDAMEMCPY2DFROMARRAY-NEXT:                     s5, 1) /*cudaMemcpyKind*/);
// CUDAMEMCPY2DFROMARRAY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DFromArrayAsync | FileCheck %s -check-prefix=CUDAMEMCPY2DFROMARRAYASYNC
// CUDAMEMCPY2DFROMARRAYASYNC: CUDA API:
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:   cudaMemcpy2DFromArrayAsync(pv /*void **/, s1 /*size_t*/,
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:                              a /*cudaArray_const_t*/, s2 /*size_t*/,
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:                              s3 /*size_t*/, s4 /*size_t*/, s5 /*size_t*/,
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:                              m /*cudaMemcpyKind*/, s /*cudaStream_t*/);
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT: Is migrated to:
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:    dpct::async_dpct_memcpy(dpct::pitched_data(pv /*void **/, s1, s1, 1) /*size_t*/,
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:                            sycl::id<3>(0, 0, 0), a->to_pitched_data() /*cudaArray_const_t*/, sycl::id<3>(s2 /*size_t*/,
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:                            s3, 0) /*size_t*/, sycl::range<3>(s4 /*size_t*/, s5, 1), dpct::automatic, *s /*cudaStream_t*/);
// CUDAMEMCPY2DFROMARRAYASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DToArray | FileCheck %s -check-prefix=CUDAMEMCPY2DTOARRAY
// CUDAMEMCPY2DTOARRAY: CUDA API:
// CUDAMEMCPY2DTOARRAY-NEXT:   cudaMemcpy2DToArray(a /*cudaArray_t*/, s1 /*size_t*/, s2 /*size_t*/,
// CUDAMEMCPY2DTOARRAY-NEXT:                       pv /*const void **/, s3 /*size_t*/, s4 /*size_t*/,
// CUDAMEMCPY2DTOARRAY-NEXT:                       s5 /*size_t*/, m /*cudaMemcpyKind*/);
// CUDAMEMCPY2DTOARRAY-NEXT: Is migrated to:
// CUDAMEMCPY2DTOARRAY-NEXT:   dpct::dpct_memcpy(a->to_pitched_data() /*cudaArray_t*/, sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/,
// CUDAMEMCPY2DTOARRAY-NEXT:                     dpct::pitched_data(pv /*const void **/, s3, s3, 1) /*size_t*/, sycl::id<3>(0, 0, 0), sycl::range<3>(s4 /*size_t*/,
// CUDAMEMCPY2DTOARRAY-NEXT:                     s5, 1) /*cudaMemcpyKind*/);
// CUDAMEMCPY2DTOARRAY-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DToArrayAsync | FileCheck %s -check-prefix=CUDAMEMCPY2DTOARRAYASYNC
// CUDAMEMCPY2DTOARRAYASYNC: CUDA API:
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:   cudaMemcpy2DToArrayAsync(a /*cudaArray_t*/, s1 /*size_t*/, s2 /*size_t*/,
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:                            pv /*const void **/, s3 /*size_t*/, s4 /*size_t*/,
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:                            s5 /*size_t*/, m /*cudaMemcpyKind*/,
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:                            s /*cudaStream_t*/);
// CUDAMEMCPY2DTOARRAYASYNC-NEXT: Is migrated to:
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:   dpct::async_dpct_memcpy(a->to_pitched_data() /*cudaArray_t*/, sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/,
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:                           dpct::pitched_data(pv /*const void **/, s3, s3, 1) /*size_t*/, sycl::id<3>(0, 0, 0), sycl::range<3>(s4 /*size_t*/,
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:                           s5, 1), dpct::automatic, *s /*cudaStream_t*/);
// CUDAMEMCPY2DTOARRAYASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy3D | FileCheck %s -check-prefix=CUDAMEMCPY3D
// CUDAMEMCPY3D: CUDA API:
// CUDAMEMCPY3D-NEXT:   cudaMemcpy3D(pm /*const cudaMemcpy3DParms **/);
// CUDAMEMCPY3D-NEXT: Is migrated to:
// CUDAMEMCPY3D-NEXT:   dpct::dpct_memcpy(pm /*const cudaMemcpy3DParms **/);
// CUDAMEMCPY3D-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy3DAsync | FileCheck %s -check-prefix=CUDAMEMCPY3DASYNC
// CUDAMEMCPY3DASYNC: CUDA API:
// CUDAMEMCPY3DASYNC-NEXT:   cudaMemcpy3DAsync(pm /*const cudaMemcpy3DParms **/, s /*cudaStream_t*/);
// CUDAMEMCPY3DASYNC-NEXT: Is migrated to:
// CUDAMEMCPY3DASYNC-NEXT:   dpct::async_dpct_memcpy(pm /*const cudaMemcpy3DParms **/, *s /*cudaStream_t*/);
// CUDAMEMCPY3DASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyAsync | FileCheck %s -check-prefix=CUDAMEMCPYASYNC
// CUDAMEMCPYASYNC: CUDA API:
// CUDAMEMCPYASYNC-NEXT:   cudaStream_t cs;
// CUDAMEMCPYASYNC-NEXT:   cudaMemcpyAsync(pv /*void **/, cpv /*const void **/, s /*size_t*/,
// CUDAMEMCPYASYNC-NEXT:                   m /*cudaMemcpyKind*/, cs /*cudaStream_t*/);
// CUDAMEMCPYASYNC-NEXT: Is migrated to:
// CUDAMEMCPYASYNC-NEXT:   dpct::queue_ptr cs;
// CUDAMEMCPYASYNC-NEXT:   cs->memcpy(pv /*void **/, cpv /*const void **/, s /*cudaStream_t*/);
// CUDAMEMCPYASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyFromSymbol | FileCheck %s -check-prefix=CUDAMEMCPYFROMSYMBOL
// CUDAMEMCPYFROMSYMBOL: CUDA API:
// CUDAMEMCPYFROMSYMBOL-NEXT:   cudaMemcpyFromSymbol(pv /*void **/, cpv /*const void **/, s1 /*size_t*/,
// CUDAMEMCPYFROMSYMBOL-NEXT:                        s2 /*size_t*/, m /*cudaMemcpyKind*/);
// CUDAMEMCPYFROMSYMBOL-NEXT: Is migrated to:
// CUDAMEMCPYFROMSYMBOL-NEXT:   dpct::get_default_queue().memcpy(pv /*void **/, cpv /*const void **/, s1 /*cudaMemcpyKind*/).wait();
// CUDAMEMCPYFROMSYMBOL-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyFromSymbolAsync | FileCheck %s -check-prefix=CUDAMEMCPYFROMSYMBOLASYNC
// CUDAMEMCPYFROMSYMBOLASYNC: CUDA API:
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:   cudaStream_t s;
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:   cudaMemcpyFromSymbolAsync(pv /*void **/, cpv /*const void **/, s1 /*size_t*/,
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:                             s2 /*size_t*/, m /*cudaMemcpyKind*/,
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:                             s /*cudaStream_t*/);
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT: Is migrated to:
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:   s->memcpy(pv /*void **/, cpv /*const void **/, s1 /*cudaStream_t*/);
// CUDAMEMCPYFROMSYMBOLASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyToSymbol | FileCheck %s -check-prefix=CUDAMEMCPYTOSYMBOL
// CUDAMEMCPYTOSYMBOL: CUDA API:
// CUDAMEMCPYTOSYMBOL-NEXT:   cudaMemcpyToSymbol(cpv1 /*const void **/, cpv2 /*const void **/,
// CUDAMEMCPYTOSYMBOL-NEXT:                      s1 /*size_t*/, s2 /*size_t*/, m /*cudaMemcpyKind*/);
// CUDAMEMCPYTOSYMBOL-NEXT: Is migrated to:
// CUDAMEMCPYTOSYMBOL-NEXT:   dpct::get_default_queue().memcpy(cpv1 /*const void **/, cpv2 /*const void **/,
// CUDAMEMCPYTOSYMBOL-NEXT:                       s1 /*cudaMemcpyKind*/).wait();
// CUDAMEMCPYTOSYMBOL-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyToSymbolAsync | FileCheck %s -check-prefix=CUDAMEMCPYTOSYMBOLASYNC
// CUDAMEMCPYTOSYMBOLASYNC: CUDA API:
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:   cudaStream_t s;
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:   cudaMemcpyToSymbolAsync(cpv1 /*const void **/, cpv2 /*const void **/,
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:                           s1 /*size_t*/, s2 /*size_t*/, m /*cudaMemcpyKind*/,
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:                           s /*cudaStream_t*/);
// CUDAMEMCPYTOSYMBOLASYNC-NEXT: Is migrated to:
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:   s->memcpy(cpv1 /*const void **/, cpv2 /*const void **/,
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:                           s1 /*cudaStream_t*/);
// CUDAMEMCPYTOSYMBOLASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset | FileCheck %s -check-prefix=CUDAMEMSET
// CUDAMEMSET: CUDA API:
// CUDAMEMSET-NEXT:   cudaMemset(pv /*void **/, i /*int*/, s /*size_t*/);
// CUDAMEMSET-NEXT: Is migrated to:
// CUDAMEMSET-NEXT:   dpct::get_default_queue().memset(pv /*void **/, i /*int*/, s /*size_t*/).wait();
// CUDAMEMSET-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset2D | FileCheck %s -check-prefix=CUDAMEMSET2D
// CUDAMEMSET2D: CUDA API:
// CUDAMEMSET2D-NEXT:   cudaMemset2D(pv /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
// CUDAMEMSET2D-NEXT:              s3 /*size_t*/);
// CUDAMEMSET2D-NEXT: Is migrated to:
// CUDAMEMSET2D-NEXT:   dpct::dpct_memset(pv /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
// CUDAMEMSET2D-NEXT:              s3 /*size_t*/);
// CUDAMEMSET2D-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset2DAsync | FileCheck %s -check-prefix=CUDAMEMSET2DASYNC
// CUDAMEMSET2DASYNC: CUDA API:
// CUDAMEMSET2DASYNC-NEXT:   cudaMemset2DAsync(pv /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
// CUDAMEMSET2DASYNC-NEXT:                     s3 /*size_t*/, s /*cudaStream_t*/);
// CUDAMEMSET2DASYNC-NEXT: Is migrated to:
// CUDAMEMSET2DASYNC-NEXT:     dpct::async_dpct_memset(pv /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
// CUDAMEMSET2DASYNC-NEXT:                     s3 /*size_t*/, *s /*cudaStream_t*/);
// CUDAMEMSET2DASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset3D | FileCheck %s -check-prefix=CUDAMEMSET3D
// CUDAMEMSET3D: CUDA API:
// CUDAMEMSET3D-NEXT:   cudaMemset3D(p /*cudaPitchedPtr*/, i /*int*/, e /*cudaExtent*/);
// CUDAMEMSET3D-NEXT: Is migrated to:
// CUDAMEMSET3D-NEXT:   dpct::dpct_memset(p /*cudaPitchedPtr*/, i /*int*/, e /*cudaExtent*/);
// CUDAMEMSET3D-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset3DAsync | FileCheck %s -check-prefix=CUDAMEMSET3DASYNC
// CUDAMEMSET3DASYNC: CUDA API:
// CUDAMEMSET3DASYNC-NEXT:   cudaMemset3DAsync(p /*cudaPitchedPtr*/, i /*int*/, e /*cudaExtent*/,
// CUDAMEMSET3DASYNC-NEXT:                     s /*cudaStream_t*/);
// CUDAMEMSET3DASYNC-NEXT: Is migrated to:
// CUDAMEMSET3DASYNC-NEXT:     dpct::async_dpct_memset(p /*cudaPitchedPtr*/, i /*int*/, e /*cudaExtent*/,
// CUDAMEMSET3DASYNC-NEXT:                     *s /*cudaStream_t*/);
// CUDAMEMSET3DASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemsetAsync | FileCheck %s -check-prefix=CUDAMEMSETASYNC
// CUDAMEMSETASYNC: CUDA API:
// CUDAMEMSETASYNC-NEXT:   cudaStream_t cs;
// CUDAMEMSETASYNC-NEXT:   cudaMemsetAsync(pv /*void **/, i /*int*/, s /*size_t*/, cs /*cudaStream_t*/);
// CUDAMEMSETASYNC-NEXT: Is migrated to:
// CUDAMEMSETASYNC-NEXT:   dpct::queue_ptr cs;
// CUDAMEMSETASYNC-NEXT:   cs->memset(pv /*void **/, i /*int*/, s /*cudaStream_t*/);
// CUDAMEMSETASYNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_cudaExtent | FileCheck %s -check-prefix=MAKE_CUDAEXTENT
// MAKE_CUDAEXTENT: CUDA API:
// MAKE_CUDAEXTENT-NEXT:   make_cudaExtent(s1 /*size_t*/, s2 /*size_t*/, s3 /*size_t*/);
// MAKE_CUDAEXTENT-NEXT: Is migrated to:
// MAKE_CUDAEXTENT-NEXT:   sycl::range<3>(s1 /*size_t*/, s2 /*size_t*/, s3 /*size_t*/);
// MAKE_CUDAEXTENT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_cudaPitchedPtr | FileCheck %s -check-prefix=MAKE_CUDAPITCHEDPTR
// MAKE_CUDAPITCHEDPTR: CUDA API:
// MAKE_CUDAPITCHEDPTR-NEXT:   make_cudaPitchedPtr(pv /*void **/, s1 /*size_t*/, s2 /*size_t*/,
// MAKE_CUDAPITCHEDPTR-NEXT:                       s3 /*size_t*/);
// MAKE_CUDAPITCHEDPTR-NEXT: Is migrated to:
// MAKE_CUDAPITCHEDPTR-NEXT:     dpct::pitched_data(pv /*void **/, s1 /*size_t*/, s2 /*size_t*/,
// MAKE_CUDAPITCHEDPTR-NEXT:                       s3 /*size_t*/);
// MAKE_CUDAPITCHEDPTR-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_cudaPos | FileCheck %s -check-prefix=MAKE_CUDAPOS
// MAKE_CUDAPOS: CUDA API:
// MAKE_CUDAPOS-NEXT:   make_cudaPos(s1 /*size_t*/, s2 /*size_t*/, s3 /*size_t*/);
// MAKE_CUDAPOS-NEXT: Is migrated to:
// MAKE_CUDAPOS-NEXT:   sycl::id<3>(s1 /*size_t*/, s2 /*size_t*/, s3 /*size_t*/);
// MAKE_CUDAPOS-EMPTY:
