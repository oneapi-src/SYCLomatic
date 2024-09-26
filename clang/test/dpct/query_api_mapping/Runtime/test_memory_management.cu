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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFree | FileCheck %s -check-prefix=CUDAFREE
// CUDAFREE: CUDA API:
// CUDAFREE-NEXT:   cudaFree(pDev /*void **/);
// CUDAFREE-NEXT: Is migrated to:
// CUDAFREE-NEXT:   dpct::dpct_free(pDev, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFreeArray | FileCheck %s -check-prefix=CUDAFREEARRAY
// CUDAFREEARRAY: CUDA API:
// CUDAFREEARRAY-NEXT:   cudaFreeArray(a /*cudaArray_t*/);
// CUDAFREEARRAY-NEXT: Is migrated to:
// CUDAFREEARRAY-NEXT:   delete a;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFreeHost | FileCheck %s -check-prefix=CUDAFREEHOST
// CUDAFREEHOST: CUDA API:
// CUDAFREEHOST-NEXT:   cudaFreeHost(pHost /*void **/);
// CUDAFREEHOST-NEXT: Is migrated to:
// CUDAFREEHOST-NEXT:   sycl::free(pHost, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaFreeMipmappedArray | FileCheck %s -check-prefix=CUDAFREEMIPMAPPEDARRAY
// CUDAFREEMIPMAPPEDARRAY: CUDA API:
// CUDAFREEMIPMAPPEDARRAY-NEXT:   cudaFreeMipmappedArray(m /*cudaMipmappedArray_t*/);
// CUDAFREEMIPMAPPEDARRAY-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDAFREEMIPMAPPEDARRAY-NEXT:   delete m;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetMipmappedArrayLevel | FileCheck %s -check-prefix=CUDAGETMIPMAPPEDARRAYLEVEL
// CUDAGETMIPMAPPEDARRAYLEVEL: CUDA API:
// CUDAGETMIPMAPPEDARRAYLEVEL-NEXT:   cudaGetMipmappedArrayLevel(a /*cudaArray_t **/,
// CUDAGETMIPMAPPEDARRAYLEVEL-NEXT:                              m /*const cudaMipmappedArray_t*/, u /*unsigned*/);
// CUDAGETMIPMAPPEDARRAYLEVEL-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDAGETMIPMAPPEDARRAYLEVEL-NEXT:   *a = m->get_mip_level(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetSymbolAddress | FileCheck %s -check-prefix=CUDAGETSYMBOLADDRESS
// CUDAGETSYMBOLADDRESS: CUDA API:
// CUDAGETSYMBOLADDRESS-NEXT:   cudaGetSymbolAddress(pDev /*void ***/, symbol /*const void **/);
// CUDAGETSYMBOLADDRESS-NEXT: Is migrated to:
// CUDAGETSYMBOLADDRESS-NEXT:   *(pDev) = symbol.get_ptr();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaGetSymbolSize | FileCheck %s -check-prefix=CUDAGETSYMBOLSIZE
// CUDAGETSYMBOLSIZE: CUDA API:
// CUDAGETSYMBOLSIZE-NEXT:   cudaGetSymbolSize(s /*size_t **/, symbol /*const void **/);
// CUDAGETSYMBOLSIZE-NEXT: Is migrated to:
// CUDAGETSYMBOLSIZE-NEXT:   *s = symbol.get_size();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostAlloc | FileCheck %s -check-prefix=CUDAHOSTALLOC
// CUDAHOSTALLOC: CUDA API:
// CUDAHOSTALLOC-NEXT:   cudaHostAlloc(pHost /*void ***/, s /*size_t*/, u /*unsigned int*/);
// CUDAHOSTALLOC-NEXT: Is migrated to:
// CUDAHOSTALLOC-NEXT:   *pHost = (void *)sycl::malloc_host(s, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostGetDevicePointer | FileCheck %s -check-prefix=CUDAHOSTGETDEVICEPOINTER
// CUDAHOSTGETDEVICEPOINTER: CUDA API:
// CUDAHOSTGETDEVICEPOINTER-NEXT:   cudaHostGetDevicePointer(pDev /*void ***/, pHost /*void **/,
// CUDAHOSTGETDEVICEPOINTER-NEXT:                            u /*unsigned int*/);
// CUDAHOSTGETDEVICEPOINTER-NEXT: Is migrated to:
// CUDAHOSTGETDEVICEPOINTER-NEXT:   *pDev = (void *)pHost;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostGetFlags | FileCheck %s -check-prefix=CUDAHOSTGETFLAGS
// CUDAHOSTGETFLAGS: CUDA API:
// CUDAHOSTGETFLAGS-NEXT:   cudaHostGetFlags(pu /*unsigned int **/, pHost /*void **/);
// CUDAHOSTGETFLAGS-NEXT: Is migrated to:
// CUDAHOSTGETFLAGS-NEXT:   *pu = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostRegister | FileCheck %s -check-prefix=CUDAHOSTREGISTER
// CUDAHOSTREGISTER: CUDA API:
// CUDAHOSTREGISTER-NEXT:   cudaHostRegister(pHost /*void **/, s /*size_t*/, u /*unsigned int*/);
// CUDAHOSTREGISTER-NEXT: The API is Removed.
// CUDAHOSTREGISTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaHostUnregister | FileCheck %s -check-prefix=CUDAHOSTUNREGISTER
// CUDAHOSTUNREGISTER: CUDA API:
// CUDAHOSTUNREGISTER-NEXT:   cudaHostUnregister(pHost /*void **/);
// CUDAHOSTUNREGISTER-NEXT: The API is Removed.
// CUDAHOSTUNREGISTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMalloc | FileCheck %s -check-prefix=CUDAMALLOC
// CUDAMALLOC: CUDA API:
// CUDAMALLOC-NEXT:   cudaMalloc(pDev /*void ***/, s /*size_t*/);
// CUDAMALLOC-NEXT: Is migrated to:
// CUDAMALLOC-NEXT:   *pDev = (void *)sycl::malloc_device(s, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMalloc3D | FileCheck %s -check-prefix=CUDAMALLOC3D
// CUDAMALLOC3D: CUDA API:
// CUDAMALLOC3D-NEXT:   cudaExtent e;
// CUDAMALLOC3D-NEXT:   cudaMalloc3D(pitch /*cudaPitchedPtr **/, e);
// CUDAMALLOC3D-NEXT: Is migrated to:
// CUDAMALLOC3D-NEXT:   sycl::range<3> e{0, 0, 0};
// CUDAMALLOC3D-NEXT:   *pitch = dpct::dpct_malloc(e);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMalloc3DArray | FileCheck %s -check-prefix=CUDAMALLOC3DARRAY
// CUDAMALLOC3DARRAY: CUDA API:
// CUDAMALLOC3DARRAY-NEXT:   const cudaChannelFormatDesc *pc;
// CUDAMALLOC3DARRAY-NEXT:   cudaMalloc3DArray(pa /*cudaArray_t **/, pc, e /*cudaExtent*/, u);
// CUDAMALLOC3DARRAY-NEXT: Is migrated to:
// CUDAMALLOC3DARRAY-NEXT:   const dpct::image_channel *pc;
// CUDAMALLOC3DARRAY-NEXT:   *pa = new dpct::image_matrix(*pc, e /*cudaExtent*/, dpct::image_type::standard);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMallocArray | FileCheck %s -check-prefix=CUDAMALLOCARRAY
// CUDAMALLOCARRAY: CUDA API:
// CUDAMALLOCARRAY-NEXT:   const cudaChannelFormatDesc *pc;
// CUDAMALLOCARRAY-NEXT:   cudaMallocArray(pa /*cudaArray_t **/, pc, s1 /*size_t*/, s2 /*size_t*/, u);
// CUDAMALLOCARRAY-NEXT: Is migrated to:
// CUDAMALLOCARRAY-NEXT:   const dpct::image_channel *pc;
// CUDAMALLOCARRAY-NEXT:   *pa = new dpct::image_matrix(*pc, sycl::range<2>(s1 /*size_t*/, s2) /*size_t*/, dpct::image_type::standard);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMallocHost | FileCheck %s -check-prefix=CUDAMALLOCHOST
// CUDAMALLOCHOST: CUDA API:
// CUDAMALLOCHOST-NEXT:   cudaMallocHost(pHost /*void ***/, s /*size_t*/);
// CUDAMALLOCHOST-NEXT: Is migrated to:
// CUDAMALLOCHOST-NEXT:   *pHost = (void *)sycl::malloc_host(s, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMallocManaged | FileCheck %s -check-prefix=CUDAMALLOCMANAGED
// CUDAMALLOCMANAGED: CUDA API:
// CUDAMALLOCMANAGED-NEXT:   cudaMallocManaged(pDev /*void ***/, s /*size_t*/, u /*unsigned int*/);
// CUDAMALLOCMANAGED-NEXT: Is migrated to:
// CUDAMALLOCMANAGED-NEXT:   *pDev = (void *)sycl::malloc_shared(s, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMallocMipmappedArray | FileCheck %s -check-prefix=CUDAMALLOCMIPMAPPEDARRAY
// CUDAMALLOCMIPMAPPEDARRAY: CUDA API:
// CUDAMALLOCMIPMAPPEDARRAY-NEXT:   cudaMallocMipmappedArray(m /*cudaMipmappedArray_t **/,
// CUDAMALLOCMIPMAPPEDARRAY-NEXT:                            d /*const cudaChannelFormatDesc **/,
// CUDAMALLOCMIPMAPPEDARRAY-NEXT:                            e /*cudaExtent*/, u1 /*unsigned*/, u2 /*unsigned*/);
// CUDAMALLOCMIPMAPPEDARRAY-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUDAMALLOCMIPMAPPEDARRAY-NEXT:   *m = new dpct::experimental::image_mem_wrapper(*d, e, sycl::ext::oneapi::experimental::image_type::mipmap, u1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMallocPitch | FileCheck %s -check-prefix=CUDAMALLOCPITCH
// CUDAMALLOCPITCH: CUDA API:
// CUDAMALLOCPITCH-NEXT:   cudaMallocPitch(pDev /*void ***/, pz /*size_t **/, s1 /*size_t*/,
// CUDAMALLOCPITCH-NEXT:                   s2 /*size_t*/);
// CUDAMALLOCPITCH-NEXT: Is migrated to:
// CUDAMALLOCPITCH-NEXT:   *pDev = dpct::dpct_malloc(*pz, s1, s2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemAdvise | FileCheck %s -check-prefix=CUDAMEMADVISE
// CUDAMEMADVISE: CUDA API:
// CUDAMEMADVISE-NEXT:   cudaMemAdvise(pDev /*const void **/, s /*size_t*/, m /*cudaMemoryAdvise*/,
// CUDAMEMADVISE-NEXT:                 i /*int*/);
// CUDAMEMADVISE-NEXT: Is migrated to:
// CUDAMEMADVISE-NEXT:   dpct::get_device(i).in_order_queue().mem_advise(pDev, s, m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemGetInfo | FileCheck %s -check-prefix=CUDAMEMGETINFO
// CUDAMEMGETINFO: CUDA API:
// CUDAMEMGETINFO-NEXT:   cudaMemGetInfo(ps1 /*size_t **/, ps2 /*size_t **/);
// CUDAMEMGETINFO-NEXT: Is migrated to:
// CUDAMEMGETINFO-NEXT:   dpct::get_current_device().get_memory_info(*ps1, *ps2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemPrefetchAsync | FileCheck %s -check-prefix=CUDAMEMPREFETCHASYNC
// CUDAMEMPREFETCHASYNC: CUDA API:
// CUDAMEMPREFETCHASYNC-NEXT:   cudaStream_t cs;
// CUDAMEMPREFETCHASYNC-NEXT:   cudaMemPrefetchAsync(pDev /*const void **/, s /*size_t*/, i /*int*/,
// CUDAMEMPREFETCHASYNC-NEXT:                        cs /*cudaStream_t*/);
// CUDAMEMPREFETCHASYNC-NEXT: Is migrated to:
// CUDAMEMPREFETCHASYNC-NEXT:   dpct::queue_ptr cs;
// CUDAMEMPREFETCHASYNC-NEXT:   cs->prefetch(pDev,s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy | FileCheck %s -check-prefix=CUDAMEMCPY
// CUDAMEMCPY: CUDA API:
// CUDAMEMCPY-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPY-NEXT:   cudaMemcpy(dst /*void **/, src /*const void **/, s /*size_t*/, m);
// CUDAMEMCPY-NEXT: Is migrated to:
// CUDAMEMCPY-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPY-NEXT:   dpct::get_in_order_queue().memcpy(dst /*void **/, src /*const void **/, s).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2D | FileCheck %s -check-prefix=CUDAMEMCPY2D
// CUDAMEMCPY2D: CUDA API:
// CUDAMEMCPY2D-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPY2D-NEXT:   cudaMemcpy2D(dst /*void **/, s1 /*size_t*/, src /*const void **/,
// CUDAMEMCPY2D-NEXT:                s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/, m);
// CUDAMEMCPY2D-NEXT: Is migrated to:
// CUDAMEMCPY2D-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPY2D-NEXT:   dpct::dpct_memcpy(dst /*void **/, s1 /*size_t*/, src /*const void **/,
// CUDAMEMCPY2D-NEXT:                s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/, m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DArrayToArray | FileCheck %s -check-prefix=CUDAMEMCPY2DARRAYTOARRAY
// CUDAMEMCPY2DARRAYTOARRAY: CUDA API:
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   cudaArray_t dst;
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   cudaArray_t src;
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   cudaMemcpy2DArrayToArray(dst, s1 /*size_t*/, s2 /*size_t*/, src,
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:                            s3 /*size_t*/, s4 /*size_t*/, s5 /*size_t*/,
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:                            s6 /*size_t*/, m);
// CUDAMEMCPY2DARRAYTOARRAY-NEXT: Is migrated to:
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   dpct::image_matrix_p dst;
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   dpct::image_matrix_p src;
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:   dpct::dpct_memcpy(dst->to_pitched_data(), sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/, src->to_pitched_data(),
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:                            sycl::id<3>(s3 /*size_t*/, s4, 0) /*size_t*/, sycl::range<3>(s5 /*size_t*/,
// CUDAMEMCPY2DARRAYTOARRAY-NEXT:                            s6, 1));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DAsync | FileCheck %s -check-prefix=CUDAMEMCPY2DASYNC
// CUDAMEMCPY2DASYNC: CUDA API:
// CUDAMEMCPY2DASYNC-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPY2DASYNC-NEXT:   cudaStream_t s;
// CUDAMEMCPY2DASYNC-NEXT:   cudaMemcpy2DAsync(dst /*void **/, s1 /*size_t*/, src /*const void **/,
// CUDAMEMCPY2DASYNC-NEXT:                     s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/, m, s);
// CUDAMEMCPY2DASYNC-NEXT: Is migrated to:
// CUDAMEMCPY2DASYNC-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPY2DASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMCPY2DASYNC-NEXT:   dpct::async_dpct_memcpy(dst /*void **/, s1 /*size_t*/, src /*const void **/,
// CUDAMEMCPY2DASYNC-NEXT:                     s2 /*size_t*/, s3 /*size_t*/, s4 /*size_t*/, m, *s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DFromArray | FileCheck %s -check-prefix=CUDAMEMCPY2DFROMARRAY
// CUDAMEMCPY2DFROMARRAY: CUDA API:
// CUDAMEMCPY2DFROMARRAY-NEXT:   cudaArray_t src;
// CUDAMEMCPY2DFROMARRAY-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPY2DFROMARRAY-NEXT:   cudaMemcpy2DFromArray(dst /*void **/, s1 /*size_t*/, src, s2 /*size_t*/,
// CUDAMEMCPY2DFROMARRAY-NEXT:                         s3 /*size_t*/, s4 /*size_t*/, s5 /*size_t*/, m);
// CUDAMEMCPY2DFROMARRAY-NEXT: Is migrated to:
// CUDAMEMCPY2DFROMARRAY-NEXT:   dpct::image_matrix_p src;
// CUDAMEMCPY2DFROMARRAY-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPY2DFROMARRAY-NEXT:   dpct::dpct_memcpy(dpct::pitched_data(dst /*void **/, s1, s1, 1) /*size_t*/, sycl::id<3>(0, 0, 0), src->to_pitched_data(), sycl::id<3>(s2 /*size_t*/,
// CUDAMEMCPY2DFROMARRAY-NEXT:                         s3, 0) /*size_t*/, sycl::range<3>(s4 /*size_t*/, s5, 1));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DFromArrayAsync | FileCheck %s -check-prefix=CUDAMEMCPY2DFROMARRAYASYNC
// CUDAMEMCPY2DFROMARRAYASYNC: CUDA API:
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:   cudaArray_t src;
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:   cudaStream_t s;
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:   cudaMemcpy2DFromArrayAsync(dst /*void **/, s1 /*size_t*/, src, s2 /*size_t*/,
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:                              s3 /*size_t*/, s4 /*size_t*/, s5 /*size_t*/,
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:                              m /*cudaMemcpyKind*/, s);
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT: Is migrated to:
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:   dpct::image_matrix_p src;
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:   dpct::async_dpct_memcpy(dpct::pitched_data(dst /*void **/, s1, s1, 1) /*size_t*/, sycl::id<3>(0, 0, 0), src->to_pitched_data(), sycl::id<3>(s2 /*size_t*/,
// CUDAMEMCPY2DFROMARRAYASYNC-NEXT:                              s3, 0) /*size_t*/, sycl::range<3>(s4 /*size_t*/, s5, 1), dpct::automatic, *s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DToArray | FileCheck %s -check-prefix=CUDAMEMCPY2DTOARRAY
// CUDAMEMCPY2DTOARRAY: CUDA API:
// CUDAMEMCPY2DTOARRAY-NEXT:   cudaArray_t dst;
// CUDAMEMCPY2DTOARRAY-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPY2DTOARRAY-NEXT:   cudaMemcpy2DToArray(dst, s1 /*size_t*/, s2 /*size_t*/, src /*const void **/,
// CUDAMEMCPY2DTOARRAY-NEXT:                       s3 /*size_t*/, s4 /*size_t*/, s5 /*size_t*/, m);
// CUDAMEMCPY2DTOARRAY-NEXT: Is migrated to:
// CUDAMEMCPY2DTOARRAY-NEXT:   dpct::image_matrix_p dst;
// CUDAMEMCPY2DTOARRAY-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPY2DTOARRAY-NEXT:   dpct::dpct_memcpy(dst->to_pitched_data(), sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/, dpct::pitched_data(src /*const void **/,
// CUDAMEMCPY2DTOARRAY-NEXT:                       s3, s3, 1) /*size_t*/, sycl::id<3>(0, 0, 0), sycl::range<3>(s4 /*size_t*/, s5, 1));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy2DToArrayAsync | FileCheck %s -check-prefix=CUDAMEMCPY2DTOARRAYASYNC
// CUDAMEMCPY2DTOARRAYASYNC: CUDA API:
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:   cudaArray_t dst;
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:   cudaStream_t s;
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:   cudaMemcpy2DToArrayAsync(dst, s1 /*size_t*/, s2 /*size_t*/,
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:                            src /*const void **/, s3 /*size_t*/, s4 /*size_t*/,
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:                            s5 /*size_t*/, m /*cudaMemcpyKind*/, s);
// CUDAMEMCPY2DTOARRAYASYNC-NEXT: Is migrated to:
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:   dpct::image_matrix_p dst;
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:   dpct::async_dpct_memcpy(dst->to_pitched_data(), sycl::id<3>(s1 /*size_t*/, s2, 0) /*size_t*/,
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:                            dpct::pitched_data(src /*const void **/, s3, s3, 1) /*size_t*/, sycl::id<3>(0, 0, 0), sycl::range<3>(s4 /*size_t*/,
// CUDAMEMCPY2DTOARRAYASYNC-NEXT:                            s5, 1), dpct::automatic, *s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy3D | FileCheck %s -check-prefix=CUDAMEMCPY3D
// CUDAMEMCPY3D: CUDA API:
// CUDAMEMCPY3D-NEXT:   const cudaMemcpy3DParms *pm;
// CUDAMEMCPY3D-NEXT:   cudaMemcpy3D(pm);
// CUDAMEMCPY3D-NEXT: Is migrated to:
// CUDAMEMCPY3D-NEXT:   const dpct::memcpy_parameter *pm;
// CUDAMEMCPY3D-NEXT:   dpct::dpct_memcpy(*pm);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpy3DAsync | FileCheck %s -check-prefix=CUDAMEMCPY3DASYNC
// CUDAMEMCPY3DASYNC: CUDA API:
// CUDAMEMCPY3DASYNC-NEXT:   const cudaMemcpy3DParms *pm;
// CUDAMEMCPY3DASYNC-NEXT:   cudaStream_t s;
// CUDAMEMCPY3DASYNC-NEXT:   cudaMemcpy3DAsync(pm, s);
// CUDAMEMCPY3DASYNC-NEXT: Is migrated to:
// CUDAMEMCPY3DASYNC-NEXT:   const dpct::memcpy_parameter *pm;
// CUDAMEMCPY3DASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMCPY3DASYNC-NEXT:   dpct::async_dpct_memcpy(*pm, *s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyAsync | FileCheck %s -check-prefix=CUDAMEMCPYASYNC
// CUDAMEMCPYASYNC: CUDA API:
// CUDAMEMCPYASYNC-NEXT:   cudaStream_t cs;
// CUDAMEMCPYASYNC-NEXT:   cudaMemcpyAsync(dst /*void **/, src /*const void **/, s /*size_t*/,
// CUDAMEMCPYASYNC-NEXT:                   m /*cudaMemcpyKind*/, cs);
// CUDAMEMCPYASYNC-NEXT: Is migrated to:
// CUDAMEMCPYASYNC-NEXT:   dpct::queue_ptr cs;
// CUDAMEMCPYASYNC-NEXT:   cs->memcpy(dst /*void **/, src /*const void **/, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyFromSymbol | FileCheck %s -check-prefix=CUDAMEMCPYFROMSYMBOL
// CUDAMEMCPYFROMSYMBOL: CUDA API:
// CUDAMEMCPYFROMSYMBOL-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPYFROMSYMBOL-NEXT:   cudaMemcpyFromSymbol(dst /*void **/, symbol /*const void **/, s1 /*size_t*/,
// CUDAMEMCPYFROMSYMBOL-NEXT:                        s2 /*size_t*/, m);
// CUDAMEMCPYFROMSYMBOL-NEXT: Is migrated to:
// CUDAMEMCPYFROMSYMBOL-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPYFROMSYMBOL-NEXT:   dpct::get_in_order_queue().memcpy(dst /*void **/, symbol /*const void **/, s1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyFromSymbolAsync | FileCheck %s -check-prefix=CUDAMEMCPYFROMSYMBOLASYNC
// CUDAMEMCPYFROMSYMBOLASYNC: CUDA API:
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:   cudaStream_t s;
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:   cudaMemcpyFromSymbolAsync(dst /*void **/, symbol /*const void **/,
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:                             s1 /*size_t*/, s2 /*size_t*/, m /*cudaMemcpyKind*/,
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:                             s);
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT: Is migrated to:
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:   s->memcpy(dst /*void **/, symbol /*const void **/,
// CUDAMEMCPYFROMSYMBOLASYNC-NEXT:                             s1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyToSymbol | FileCheck %s -check-prefix=CUDAMEMCPYTOSYMBOL
// CUDAMEMCPYTOSYMBOL: CUDA API:
// CUDAMEMCPYTOSYMBOL-NEXT:   cudaMemcpyKind m;
// CUDAMEMCPYTOSYMBOL-NEXT:   cudaMemcpyToSymbol(symbol /*const void **/, src /*const void **/,
// CUDAMEMCPYTOSYMBOL-NEXT:                      s1 /*size_t*/, s2 /*size_t*/, m);
// CUDAMEMCPYTOSYMBOL-NEXT: Is migrated to:
// CUDAMEMCPYTOSYMBOL-NEXT:   dpct::memcpy_direction m;
// CUDAMEMCPYTOSYMBOL-NEXT:   dpct::get_in_order_queue().memcpy(symbol /*const void **/, src /*const void **/,
// CUDAMEMCPYTOSYMBOL-NEXT:                      s1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemcpyToSymbolAsync | FileCheck %s -check-prefix=CUDAMEMCPYTOSYMBOLASYNC
// CUDAMEMCPYTOSYMBOLASYNC: CUDA API:
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:   cudaStream_t s;
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:   cudaMemcpyToSymbolAsync(symbol /*const void **/, src /*const void **/,
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:                           s1 /*size_t*/, s2 /*size_t*/, m /*cudaMemcpyKind*/,
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:                           s);
// CUDAMEMCPYTOSYMBOLASYNC-NEXT: Is migrated to:
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:   s->memcpy(symbol /*const void **/, src /*const void **/,
// CUDAMEMCPYTOSYMBOLASYNC-NEXT:                           s1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset | FileCheck %s -check-prefix=CUDAMEMSET
// CUDAMEMSET: CUDA API:
// CUDAMEMSET-NEXT:   cudaMemset(pDev /*void **/, i /*int*/, s /*size_t*/);
// CUDAMEMSET-NEXT: Is migrated to:
// CUDAMEMSET-NEXT:   dpct::get_in_order_queue().memset(pDev /*void **/, i /*int*/, s /*size_t*/).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset2D | FileCheck %s -check-prefix=CUDAMEMSET2D
// CUDAMEMSET2D: CUDA API:
// CUDAMEMSET2D-NEXT:   cudaMemset2D(pDev /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
// CUDAMEMSET2D-NEXT:              s3 /*size_t*/);
// CUDAMEMSET2D-NEXT: Is migrated to:
// CUDAMEMSET2D-NEXT:   dpct::dpct_memset(pDev /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
// CUDAMEMSET2D-NEXT:              s3 /*size_t*/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset2DAsync | FileCheck %s -check-prefix=CUDAMEMSET2DASYNC
// CUDAMEMSET2DASYNC: CUDA API:
// CUDAMEMSET2DASYNC-NEXT:   cudaStream_t s;
// CUDAMEMSET2DASYNC-NEXT:   cudaMemset2DAsync(pDev /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
// CUDAMEMSET2DASYNC-NEXT:                     s3 /*size_t*/, s);
// CUDAMEMSET2DASYNC-NEXT: Is migrated to:
// CUDAMEMSET2DASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMSET2DASYNC-NEXT:   dpct::async_dpct_memset(pDev /*void **/, s1 /*size_t*/, i /*int*/, s2 /*size_t*/,
// CUDAMEMSET2DASYNC-NEXT:                     s3 /*size_t*/, *s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset3D | FileCheck %s -check-prefix=CUDAMEMSET3D
// CUDAMEMSET3D: CUDA API:
// CUDAMEMSET3D-NEXT:   cudaPitchedPtr p;
// CUDAMEMSET3D-NEXT:   cudaExtent e;
// CUDAMEMSET3D-NEXT:   cudaMemset3D(p, i /*int*/, e);
// CUDAMEMSET3D-NEXT: Is migrated to:
// CUDAMEMSET3D-NEXT:   dpct::pitched_data p;
// CUDAMEMSET3D-NEXT:   sycl::range<3> e{0, 0, 0};
// CUDAMEMSET3D-NEXT:   dpct::dpct_memset(p, i /*int*/, e);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemset3DAsync | FileCheck %s -check-prefix=CUDAMEMSET3DASYNC
// CUDAMEMSET3DASYNC: CUDA API:
// CUDAMEMSET3DASYNC-NEXT:   cudaPitchedPtr p;
// CUDAMEMSET3DASYNC-NEXT:   cudaExtent e;
// CUDAMEMSET3DASYNC-NEXT:   cudaStream_t s;
// CUDAMEMSET3DASYNC-NEXT:   cudaMemset3DAsync(p, i /*int*/, e, s);
// CUDAMEMSET3DASYNC-NEXT: Is migrated to:
// CUDAMEMSET3DASYNC-NEXT:   dpct::pitched_data p;
// CUDAMEMSET3DASYNC-NEXT:   sycl::range<3> e{0, 0, 0};
// CUDAMEMSET3DASYNC-NEXT:   dpct::queue_ptr s;
// CUDAMEMSET3DASYNC-NEXT:   dpct::async_dpct_memset(p, i /*int*/, e, *s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cudaMemsetAsync | FileCheck %s -check-prefix=CUDAMEMSETASYNC
// CUDAMEMSETASYNC: CUDA API:
// CUDAMEMSETASYNC-NEXT:   cudaStream_t cs;
// CUDAMEMSETASYNC-NEXT:   cudaMemsetAsync(pDev /*void **/, i /*int*/, s /*size_t*/, cs);
// CUDAMEMSETASYNC-NEXT: Is migrated to:
// CUDAMEMSETASYNC-NEXT:   dpct::queue_ptr cs;
// CUDAMEMSETASYNC-NEXT:   cs->memset(pDev /*void **/, i /*int*/, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_cudaExtent | FileCheck %s -check-prefix=MAKE_CUDAEXTENT
// MAKE_CUDAEXTENT: CUDA API:
// MAKE_CUDAEXTENT-NEXT:   make_cudaExtent(s1 /*size_t*/, s2 /*size_t*/, s3 /*size_t*/);
// MAKE_CUDAEXTENT-NEXT: Is migrated to:
// MAKE_CUDAEXTENT-NEXT:   sycl::range<3>(s1, s2, s3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_cudaPitchedPtr | FileCheck %s -check-prefix=MAKE_CUDAPITCHEDPTR
// MAKE_CUDAPITCHEDPTR: CUDA API:
// MAKE_CUDAPITCHEDPTR-NEXT:   make_cudaPitchedPtr(ptr /*void **/, s1 /*size_t*/, s2 /*size_t*/,
// MAKE_CUDAPITCHEDPTR-NEXT:                       s3 /*size_t*/);
// MAKE_CUDAPITCHEDPTR-NEXT: Is migrated to:
// MAKE_CUDAPITCHEDPTR-NEXT:   dpct::pitched_data(ptr, s1, s2, s3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_cudaPos | FileCheck %s -check-prefix=MAKE_CUDAPOS
// MAKE_CUDAPOS: CUDA API:
// MAKE_CUDAPOS-NEXT:   make_cudaPos(s1 /*size_t*/, s2 /*size_t*/, s3 /*size_t*/);
// MAKE_CUDAPOS-NEXT: Is migrated to:
// MAKE_CUDAPOS-NEXT:   sycl::id<3>(s1, s2, s3);
