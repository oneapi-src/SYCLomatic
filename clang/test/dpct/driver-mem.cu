// RUN: dpct --format-range=none -out-root %T/driver-mem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-mem/driver-mem.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/driver-mem/driver-mem.dp.cpp -o %T/driver-mem/driver-mem.dp.o %}

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define CALL(x) x

void cuCheckError(CUresult err) {
}

int main(){
    size_t result1, result2;
    int size = 32;
    size_t count = 32;
    float* f_A;
    CUresult r;
    // CHECK: f_A = (float *)sycl::malloc_host(size, q_ct1);
    cuMemHostAlloc((void **)&f_A, size, CU_MEMHOSTALLOC_DEVICEMAP);


    CUdeviceptr p1;
    [&p1]() {
        //CHECK:p1 = 0;
        p1 = 0;
    }();

    CUdeviceptr p2;
    [&]() {
        //CHECK:p2 = 0;
        p2 = 0;
    }();

    // CHECK: dpct::device_ptr f_D = 0;
    CUdeviceptr f_D = 0;
    // CHECK: dpct::device_ptr f_D2 = 0;
    CUdeviceptr f_D2 = 0;
    // CHECK: f_D = (dpct::device_ptr)sycl::malloc_device(size, q_ct1);
    cuMemAlloc(&f_D, size);

    // CHECK: dpct::queue_ptr stream;
    CUstream stream;
    // CHECK: stream->memcpy(f_D, f_A, size);
    cuMemcpyHtoDAsync(f_D, f_A, size, stream);
    // CHECK: q_ct1.memcpy(f_D, f_A, size);
    cuMemcpyHtoDAsync(f_D, f_A, size, 0);
    // CHECK: CALL(q_ct1.memcpy(f_D, f_A, size).wait());
    CALL(cuMemcpyHtoD(f_D, f_A, size));

    // CHECK: stream->memcpy(f_A, f_D, size);
    cuMemcpyDtoHAsync(f_A, f_D, size, stream);
    // CHECK: q_ct1.memcpy(f_A, f_D, size);
    cuMemcpyDtoHAsync(f_A, f_D, size, 0);
    // CHECK: q_ct1.memcpy(f_A, f_D, size).wait();
    cuMemcpyDtoH(f_A, f_D, size);

    // CHECK: stream->memcpy(f_D, f_D2, size);
    cuMemcpyDtoDAsync(f_D, f_D2, size, stream);
    // CHECK: r = DPCT_CHECK_ERROR(stream->memcpy(f_D, f_D2, size));
    r = cuMemcpyDtoDAsync(f_D, f_D2, size, stream);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size);
    cuMemcpyDtoDAsync(f_D, f_D2, size, 0);
    // CHECK: r = DPCT_CHECK_ERROR(q_ct1.memcpy(f_D, f_D2, size));
    r = cuMemcpyDtoDAsync(f_D, f_D2, size, 0);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size).wait();
    cuMemcpyDtoD(f_D, f_D2, size);
    // CHECK: r = DPCT_CHECK_ERROR(q_ct1.memcpy(f_D, f_D2, size).wait());
    r = cuMemcpyDtoD(f_D, f_D2, size);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size).wait();
    cuMemcpy(f_D, f_D2, size);
    // CHECK: CALL(q_ct1.memcpy(f_D, f_D2, size).wait());
    CALL(cuMemcpy(f_D, f_D2, size));
    // CHECK: r = DPCT_CHECK_ERROR(q_ct1.memcpy(f_D, f_D2, size).wait());
    r = cuMemcpy(f_D, f_D2, size);

    // CHECK: stream->memcpy(f_D, f_D2, size);
    cuMemcpyAsync(f_D, f_D2, size, stream);
    // CHECK: CALL(stream->memcpy(f_D, f_D2, size));
    CALL(cuMemcpyAsync(f_D, f_D2, size, stream));
    // CHECK: r = DPCT_CHECK_ERROR(stream->memcpy(f_D, f_D2, size));
    r = cuMemcpyAsync(f_D, f_D2, size, stream);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size);
    cuMemcpyAsync(f_D, f_D2, size, 0);
    // CHECK: CALL(q_ct1.memcpy(f_D, f_D2, size));
    CALL(cuMemcpyAsync(f_D, f_D2, size, 0));
    // CHECK: r = DPCT_CHECK_ERROR(q_ct1.memcpy(f_D, f_D2, size));
    r = cuMemcpyAsync(f_D, f_D2, size, 0);
    unsigned int v32 = 50000;
    unsigned short v16 = 20000;
    unsigned char v8 = (unsigned char) 200;
    //CHECK: dpct::dpct_memset_d32(f_D, v32, size);
    //CHECK-NEXT: dpct::dpct_memset_d16(f_D, v16, size * 2);
    //CHECK-NEXT: dpct::dpct_memset(f_D, v8, size * 4);
    //CHECK-NEXT: dpct::async_dpct_memset_d32(f_D, v32, size, *stream);
    //CHECK-NEXT: dpct::async_dpct_memset_d16(f_D, v16, size * 2, *stream);
    //CHECK-NEXT: dpct::async_dpct_memset(f_D, v8, size * 4, *stream);
    //CHECK-NEXT: dpct::dpct_memset_d32(f_D, 1, v32, 4, 6);
    //CHECK-NEXT: dpct::dpct_memset_d16(f_D, 1, v16, 4 * 2, 6);
    //CHECK-NEXT: dpct::dpct_memset(f_D, 1, v8, 4 * 4, 6);
    //CHECK-NEXT: dpct::async_dpct_memset_d32(f_D, 1, v32, 4, 6, *stream);
    //CHECK-NEXT: dpct::async_dpct_memset_d16(f_D, 1, v16, 4 * 2, 6, *stream);
    //CHECK-NEXT: dpct::async_dpct_memset(f_D, 1, v8, 4 * 4, 6, *stream);
    cuMemsetD32(f_D, v32, size);
    cuMemsetD16(f_D, v16, size * 2);
    cuMemsetD8(f_D, v8, size * 4);
    cuMemsetD32Async(f_D, v32, size, stream);
    cuMemsetD16Async(f_D, v16, size * 2, stream);
    cuMemsetD8Async(f_D, v8, size * 4, stream);
    cuMemsetD2D32(f_D, 1, v32, 4, 6);
    cuMemsetD2D16(f_D, 1, v16, 4 * 2, 6);
    cuMemsetD2D8(f_D, 1, v8, 4 * 4, 6);
    cuMemsetD2D32Async(f_D, 1, v32, 4, 6, stream);
    cuMemsetD2D16Async(f_D, 1, v16, 4 * 2, 6, stream);
    cuMemsetD2D8Async(f_D, 1, v8, 4 * 4, 6, stream);

    // CHECK: dpct::mem_cpy_param_wrapper cpy;
    CUDA_MEMCPY2D cpy;
    //
    cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
    // CHECK: cpy.set_ptr_to_data(f_A);
    cpy.dstHost = f_A;
    // CHECK: cpy.set_p_to_data(20);
    cpy.dstPitch = 20;
    // CHECK: cpy.set_y_to_pos(10);
    cpy.dstY = 10;
    // CHECK: cpy.set_x_to_pos(15);
    cpy.dstXInBytes = 15;

    //
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    // CHECK: cpy.set_ptr_from_data(f_D);
    cpy.srcDevice = f_D;
    // CHECK: cpy.set_p_from_data(20);
    cpy.srcPitch = 20;
    // CHECK: cpy.set_y_from_pos(10);
    cpy.srcY = 10;
    // CHECK: cpy.set_x_from_pos(15);
    cpy.srcXInBytes = 15;

    // CHECK: cpy.set_x(4);
    cpy.WidthInBytes = 4;
    // CHECK: cpy.set_y(7);
    cpy.Height = 7;

    // CHECK: dpct::dpct_memcpy(&cpy);
    cuMemcpy2D(&cpy);
    // CHECK: dpct::async_dpct_memcpy(&cpy, *stream);
    cuMemcpy2DAsync(&cpy, stream);

    CUdeviceptr devicePtr;

    CUresult cu_err;

    CUdeviceptr cuDevPtr;

    CUdevice cudevice;
    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: int advise = 0;
    CUmem_advise advise = CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION;

    // CHECK: dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, advise);
    cuMemAdvise(devicePtr, count, advise, cudevice);

    // CHECK: cuCheckError(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, advise)));
    cuCheckError(cuMemAdvise(devicePtr, count, advise, cudevice));

    // CHECK: cu_err = DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, advise));
    cu_err = cuMemAdvise(devicePtr, count, advise, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, 0);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, 0)));
    cuCheckError(cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, 0)));
    cuCheckError(cuMemAdvise(devicePtr, count, (CUmem_advise)1, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, 0)));
    cuCheckError(cuMemAdvise(devicePtr, count, CUmem_advise(1), cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, 0)));
    cuCheckError(cuMemAdvise(devicePtr, count, static_cast<CUmem_advise>(1), cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cu_err = DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, 0));
    cu_err = cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().mem_advise(devicePtr, count, 0);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: dpct::cpu_device().in_order_queue().mem_advise(devicePtr, count, 0);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, CU_DEVICE_CPU);


    CUdeviceptr devPtr;
    CUresult curesult;
    // CHECK: stream->prefetch(devPtr, 100);
    cuMemPrefetchAsync (devPtr, 100, cudevice, stream);
    // CHECK: (*&stream)->prefetch(devPtr, 100);
    cuMemPrefetchAsync (devPtr, 100, cudevice, *&stream);
    // CHECK: curesult = DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().prefetch(devPtr, 100));
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, NULL);
    // CHECK: dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().prefetch(devPtr, 100);
    cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamPerThread);
    // CHECK: curesult = DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().prefetch(devPtr, 100));
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamDefault);
    // CHECK: curesult = DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().prefetch(devPtr, 100));
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamLegacy);
    // CHECK: curesult = DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().prefetch(devPtr, 100));
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamPerThread);
    // CHECK: cuCheckError(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().prefetch(devPtr, 100)));
    cuCheckError(cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamDefault));
    // CHECK: cuCheckError(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().prefetch(devPtr, 100)));
    cuCheckError(cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamLegacy));
    // CHECK: cuCheckError(DPCT_CHECK_ERROR(dpct::dev_mgr::instance().get_device(cudevice).in_order_queue().prefetch(devPtr, 100)));
    cuCheckError(cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamPerThread));

    // CHECK: dpct::mem_cpy_param_wrapper cpy2;
    CUDA_MEMCPY3D cpy2;

    CUarray ca;
    //
    cpy2.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    // CHECK: cpy2.set_to_image_data(ca);
    cpy2.dstArray = ca;
    // CHECK: cpy2.set_p_to_data(5);
    cpy2.dstPitch = 5;
    // CHECK: cpy2.set_y_to_data(4);
    cpy2.dstHeight = 4;
    // CHECK: cpy2.set_y_to_pos(3);
    cpy2.dstY = 3;
    // CHECK: cpy2.set_z_to_pos(2);
    cpy2.dstZ = 2;
    // CHECK: cpy2.set_x_to_pos(1);
    cpy2.dstXInBytes = 1;
    //
    cpy2.dstLOD = 0;

    //
    cpy2.srcMemoryType = CU_MEMORYTYPE_HOST;
    // CHECK: cpy2.set_ptr_from_data(f_A);
    cpy2.srcHost = f_A;
    // CHECK: cpy2.set_p_from_data(5);
    cpy2.srcPitch = 5;
    // CHECK: cpy2.set_y_from_data(4);
    cpy2.srcHeight = 4;
    // CHECK: cpy2.set_y_from_pos(3);
    cpy2.srcY = 3;
    // CHECK: cpy2.set_z_from_pos(2);
    cpy2.srcZ = 2;
    // CHECK: cpy2.set_x_from_pos(1);
    cpy2.srcXInBytes = 1;
    //
    cpy2.srcLOD = 0;

    // CHECK: cpy2.set_x(3);
    cpy2.WidthInBytes = 3;
    // CHECK: cpy2.set_y(2);
    cpy2.Height = 2;
    // CHECK: cpy2.set_z(1);
    cpy2.Depth = 1;

    // CHECK: dpct::dpct_memcpy(&cpy2);
    cuMemcpy3D(&cpy2);

    CUstream cs;
    // CHECK: dpct::async_dpct_memcpy(&cpy2, *cs);
    cuMemcpy3DAsync(&cpy2, cs);

    float *h_A = (float *)malloc(100);
    // CHECK:sycl::free(h_A, q_ct1);
    cuMemFreeHost(h_A);
    // CHECK:sycl::free(f_D, q_ct1);
    cuMemFree(f_D);

    unsigned int flags;
    int host;


    // CHECK: flags = 0;
    cuMemHostGetFlags(&flags, &host);
    // CHECK: cuCheckError(DPCT_CHECK_ERROR(flags = 0));
    cuCheckError(cuMemHostGetFlags(&flags, &host));

    // CHECK:  /*
    // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuMemHostRegister was removed because SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.
    // CHECK-NEXT: */
    cuMemHostRegister(h_A, count, flags);
    // CHECK:  /*
    // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuMemHostRegister was replaced with 0 because SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(0);
    cuCheckError(cuMemHostRegister(h_A, count, flags));


    // CHECK:  /*
    // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuMemHostUnregister was removed because SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.
    // CHECK-NEXT: */
    cuMemHostUnregister(h_A);

    // CHECK:  /*
    // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuMemHostUnregister was replaced with 0 because SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.
    // CHECK-NEXT: */
    // CHECK-NEXT:cuCheckError(0);
    cuCheckError(cuMemHostUnregister(h_A));
    return 0;
}
