/// Memory Management

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuArray3DCreate | FileCheck %s -check-prefix=CUARRAY3DCREATE
// CUARRAY3DCREATE: CUDA API:
// CUARRAY3DCREATE-NEXT:   cuArray3DCreate(pa /*CUarray **/, pd /*const CUDA_ARRAY3D_DESCRIPTOR **/);
// CUARRAY3DCREATE-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUARRAY3DCREATE-NEXT:   *pa = new dpct::experimental::image_mem_wrapper(pd);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuArrayCreate | FileCheck %s -check-prefix=CUARRAYCREATE
// CUARRAYCREATE: CUDA API:
// CUARRAYCREATE-NEXT:   cuArrayCreate(pa /*CUarray **/, pd /*const CUDA_ARRAY_DESCRIPTOR **/);
// CUARRAYCREATE-NEXT: Is migrated to (with the option --use-experimental-features=bindless_images):
// CUARRAYCREATE-NEXT:   *pa = new dpct::experimental::image_mem_wrapper(pd);

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
// CUMEMHOSTREGISTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemHostUnregister | FileCheck %s -check-prefix=CUMEMHOSTUNREGISTER
// CUMEMHOSTUNREGISTER: CUDA API:
// CUMEMHOSTUNREGISTER-NEXT:   cuMemHostUnregister(pHost /*void **/);
// CUMEMHOSTUNREGISTER-NEXT: The API is Removed.
// CUMEMHOSTUNREGISTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy | FileCheck %s -check-prefix=CUMEMCPY
// CUMEMCPY: CUDA API:
// CUMEMCPY-NEXT:   cuMemcpy(d1 /*CUdeviceptr*/, d2 /*CUdeviceptr*/, s /*size_t*/);
// CUMEMCPY-NEXT: Is migrated to:
// CUMEMCPY-NEXT:   dpct::get_in_order_queue().memcpy(d1, d2, s).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy2D | FileCheck %s -check-prefix=CUMEMCPY2D
// CUMEMCPY2D: CUDA API:
// CUMEMCPY2D-NEXT:   const CUDA_MEMCPY2D *c;
// CUMEMCPY2D-NEXT:   cuMemcpy2D(c);
// CUMEMCPY2D-NEXT: Is migrated to:
// CUMEMCPY2D-NEXT:   const dpct::memcpy_parameter *c;
// CUMEMCPY2D-NEXT:   dpct::dpct_memcpy(*c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy2DAsync | FileCheck %s -check-prefix=CUMEMCPY2DASYNC
// CUMEMCPY2DASYNC: CUDA API:
// CUMEMCPY2DASYNC-NEXT:   const CUDA_MEMCPY2D *c;
// CUMEMCPY2DASYNC-NEXT:   CUstream cs;
// CUMEMCPY2DASYNC-NEXT:   cuMemcpy2DAsync(c, cs);
// CUMEMCPY2DASYNC-NEXT: Is migrated to:
// CUMEMCPY2DASYNC-NEXT:   const dpct::memcpy_parameter *c;
// CUMEMCPY2DASYNC-NEXT:   dpct::queue_ptr cs;
// CUMEMCPY2DASYNC-NEXT:   dpct::async_dpct_memcpy(*c, *cs)

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy3D | FileCheck %s -check-prefix=CUMEMCPY3D
// CUMEMCPY3D: CUDA API:
// CUMEMCPY3D-NEXT:   const CUDA_MEMCPY3D *c;
// CUMEMCPY3D-NEXT:   cuMemcpy3D(c);
// CUMEMCPY3D-NEXT: Is migrated to:
// CUMEMCPY3D-NEXT:   const dpct::memcpy_parameter *c;
// CUMEMCPY3D-NEXT:   dpct::dpct_memcpy(*c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy3DAsync | FileCheck %s -check-prefix=CUMEMCPY3DASYNC
// CUMEMCPY3DASYNC: CUDA API:
// CUMEMCPY3DASYNC-NEXT:   const CUDA_MEMCPY3D *c;
// CUMEMCPY3DASYNC-NEXT:   CUstream cs;
// CUMEMCPY3DASYNC-NEXT:   cuMemcpy3DAsync(c, cs);
// CUMEMCPY3DASYNC-NEXT: Is migrated to:
// CUMEMCPY3DASYNC-NEXT:   const dpct::memcpy_parameter *c;
// CUMEMCPY3DASYNC-NEXT:   dpct::queue_ptr cs;
// CUMEMCPY3DASYNC-NEXT:   dpct::async_dpct_memcpy(*c, *cs)

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy3DPeer | FileCheck %s -check-prefix=CUMEMCPY3DPEER
// CUMEMCPY3DPEER: CUDA API:
// CUMEMCPY3DPEER-NEXT:   cuMemcpy3DPeer(c /*CUDA_MEMCPY3D_PEER **/);
// CUMEMCPY3DPEER-NEXT: Is migrated to:
// CUMEMCPY3DPEER-NEXT:   dpct::dpct_memcpy(*c /*CUDA_MEMCPY3D_PEER **/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpy3DPeerAsync | FileCheck %s -check-prefix=CUMEMCPY3DPEERASYNC
// CUMEMCPY3DPEERASYNC: CUDA API:
// CUMEMCPY3DPEERASYNC-NEXT:   cuMemcpy3DPeerAsync(c /*CUDA_MEMCPY3D_PEER **/, cs /*CUstream*/);
// CUMEMCPY3DPEERASYNC-NEXT: Is migrated to:
// CUMEMCPY3DPEERASYNC-NEXT:   dpct::async_dpct_memcpy(*c /*CUDA_MEMCPY3D_PEER **/, *cs /*CUstream*/)

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyPeer | FileCheck %s -check-prefix=CUMEMCPYPEER
// CUMEMCPYPEER: CUDA API:
// CUMEMCPYPEER-NEXT:   cuMemcpyPeer(pd1 /*CUdeviceptr*/, c1 /*CUcontext*/, pd2 /*CUdeviceptr*/,
// CUMEMCPYPEER-NEXT:                c2 /*CUcontext*/, s /*size_t*/);
// CUMEMCPYPEER-NEXT: Is migrated to:
// CUMEMCPYPEER-NEXT:   dpct::dpct_memcpy(pd1 /*CUdeviceptr*/, c1 /*CUcontext*/, pd2 /*CUdeviceptr*/,
// CUMEMCPYPEER-NEXT:                c2 /*CUcontext*/, s /*size_t*/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemcpyPeerAsync | FileCheck %s -check-prefix=CUMEMCPYPEERASYNC
// CUMEMCPYPEERASYNC: CUDA API:
// CUMEMCPYPEERASYNC-NEXT:   cuMemcpyPeerAsync(pd1 /*CUdeviceptr*/, c1 /*CUcontext*/, pd2 /*CUdeviceptr*/,
// CUMEMCPYPEERASYNC-NEXT:                     c2 /*CUcontext*/, s /*size_t*/, cs /*CUstream*/);
// CUMEMCPYPEERASYNC-NEXT: Is migrated to:
// CUMEMCPYPEERASYNC-NEXT:   dpct::async_dpct_memcpy(pd1 /*CUdeviceptr*/, c1 /*CUcontext*/, pd2 /*CUdeviceptr*/,
// CUMEMCPYPEERASYNC-NEXT:                     c2 /*CUcontext*/, s /*size_t*/, *cs /*CUstream*/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD16 | FileCheck %s -check-prefix=CUMEMSETD16
// CUMEMSETD16: CUDA API:
// CUMEMSETD16-NEXT:   cuMemsetD16(d /*CUdeviceptr*/, us /*unsigned short*/, s /*size_t*/);
// CUMEMSETD16-NEXT: Is migrated to:
// CUMEMSETD16-NEXT:   dpct::dpct_memset_d16(d, us, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD16Async | FileCheck %s -check-prefix=CUMEMSETD16ASYNC
// CUMEMSETD16ASYNC: CUDA API:
// CUMEMSETD16ASYNC-NEXT:   cuMemsetD16Async(d /*CUdeviceptr*/, us /*unsigned short*/, s /*size_t*/,
// CUMEMSETD16ASYNC-NEXT:                    cs /*CUstream*/);
// CUMEMSETD16ASYNC-NEXT: Is migrated to:
// CUMEMSETD16ASYNC-NEXT:   dpct::async_dpct_memset_d16(d, us, s, *cs);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD2D16 | FileCheck %s -check-prefix=CUMEMSETD2D16
// CUMEMSETD2D16: CUDA API:
// CUMEMSETD2D16-NEXT:   cuMemsetD2D16(d /*CUdeviceptr*/, s1 /*size_t*/, us /*unsigned short*/,
// CUMEMSETD2D16-NEXT:                 s2 /*size_t*/, s3 /*size_t*/);
// CUMEMSETD2D16-NEXT: Is migrated to:
// CUMEMSETD2D16-NEXT:   dpct::dpct_memset_d16(d, s1, us, s2, s3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD2D16Async | FileCheck %s -check-prefix=CUMEMSETD2D16ASYNC
// CUMEMSETD2D16ASYNC: CUDA API:
// CUMEMSETD2D16ASYNC-NEXT:   cuMemsetD2D16Async(d /*CUdeviceptr*/, s1 /*size_t*/, us /*unsigned short*/,
// CUMEMSETD2D16ASYNC-NEXT:                      s2 /*size_t*/, s3 /*size_t*/, cs /*CUstream*/);
// CUMEMSETD2D16ASYNC-NEXT: Is migrated to:
// CUMEMSETD2D16ASYNC-NEXT:   dpct::async_dpct_memset_d16(d, s1, us, s2, s3, *cs);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD2D32 | FileCheck %s -check-prefix=CUMEMSETD2D32
// CUMEMSETD2D32: CUDA API:
// CUMEMSETD2D32-NEXT:   cuMemsetD2D32(d /*CUdeviceptr*/, s1 /*size_t*/, u /*unsigned*/, s2 /*size_t*/,
// CUMEMSETD2D32-NEXT:                 s3 /*size_t*/);
// CUMEMSETD2D32-NEXT: Is migrated to:
// CUMEMSETD2D32-NEXT:   dpct::dpct_memset_d32(d, s1, u, s2, s3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD2D32Async | FileCheck %s -check-prefix=CUMEMSETD2D32ASYNC
// CUMEMSETD2D32ASYNC: CUDA API:
// CUMEMSETD2D32ASYNC-NEXT:   cuMemsetD2D32Async(d /*CUdeviceptr*/, s1 /*size_t*/, u /*unsigned*/,
// CUMEMSETD2D32ASYNC-NEXT:                      s2 /*size_t*/, s3 /*size_t*/, cs /*CUstream*/);
// CUMEMSETD2D32ASYNC-NEXT: Is migrated to:
// CUMEMSETD2D32ASYNC-NEXT:   dpct::async_dpct_memset_d32(d, s1, u, s2, s3, *cs);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD2D8 | FileCheck %s -check-prefix=CUMEMSETD2D8
// CUMEMSETD2D8: CUDA API:
// CUMEMSETD2D8-NEXT:   cuMemsetD2D8(d /*CUdeviceptr*/, s1 /*size_t*/, uc /*unsigned char*/,
// CUMEMSETD2D8-NEXT:                s2 /*size_t*/, s3 /*size_t*/);
// CUMEMSETD2D8-NEXT: Is migrated to:
// CUMEMSETD2D8-NEXT:   dpct::dpct_memset(d, s1, uc, s2, s3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD2D8Async | FileCheck %s -check-prefix=CUMEMSETD2D8ASYNC
// CUMEMSETD2D8ASYNC: CUDA API:
// CUMEMSETD2D8ASYNC-NEXT:   cuMemsetD2D8Async(d /*CUdeviceptr*/, s1 /*size_t*/, uc /*unsigned char*/,
// CUMEMSETD2D8ASYNC-NEXT:                     s2 /*size_t*/, s3 /*size_t*/, cs /*CUstream*/);
// CUMEMSETD2D8ASYNC-NEXT: Is migrated to:
// CUMEMSETD2D8ASYNC-NEXT:   dpct::async_dpct_memset(d, s1, uc, s2, s3, *cs);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD32 | FileCheck %s -check-prefix=CUMEMSETD32
// CUMEMSETD32: CUDA API:
// CUMEMSETD32-NEXT:   cuMemsetD32(d /*CUdeviceptr*/, u /*unsigned*/, s /*size_t*/);
// CUMEMSETD32-NEXT: Is migrated to:
// CUMEMSETD32-NEXT:   dpct::dpct_memset_d32(d, u, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD32Async | FileCheck %s -check-prefix=CUMEMSETD32ASYNC
// CUMEMSETD32ASYNC: CUDA API:
// CUMEMSETD32ASYNC-NEXT:   cuMemsetD32Async(d /*CUdeviceptr*/, u /*unsigned*/, s /*size_t*/,
// CUMEMSETD32ASYNC-NEXT:                    cs /*CUstream*/);
// CUMEMSETD32ASYNC-NEXT: Is migrated to:
// CUMEMSETD32ASYNC-NEXT:   dpct::async_dpct_memset_d32(d, u, s, *cs);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD8 | FileCheck %s -check-prefix=CUMEMSETD8
// CUMEMSETD8: CUDA API:
// CUMEMSETD8-NEXT:   cuMemsetD8(d /*CUdeviceptr*/, uc /*unsigned char*/, s /*size_t*/);
// CUMEMSETD8-NEXT: Is migrated to:
// CUMEMSETD8-NEXT:   dpct::dpct_memset(d, uc, s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuMemsetD8Async | FileCheck %s -check-prefix=CUMEMSETD8ASYNC
// CUMEMSETD8ASYNC: CUDA API:
// CUMEMSETD8ASYNC-NEXT:   cuMemsetD8Async(d /*CUdeviceptr*/, uc /*unsigned char*/, s /*size_t*/,
// CUMEMSETD8ASYNC-NEXT:                   cs /*CUstream*/);
// CUMEMSETD8ASYNC-NEXT: Is migrated to:
// CUMEMSETD8ASYNC-NEXT:   dpct::async_dpct_memset(d, uc, s, *cs);
