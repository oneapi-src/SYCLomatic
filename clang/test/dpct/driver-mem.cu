// RUN: dpct --format-range=none -out-root %T/driver-mem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-mem/driver-mem.dp.cpp %s

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


    // CHECK: char * f_D = 0;
    CUdeviceptr f_D = 0;
    // CHECK: char * f_D2 = 0;
    CUdeviceptr f_D2 = 0;
    // CHECK: f_D = (char *)sycl::malloc_device(size, q_ct1);
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
    // CHECK: r = (stream->memcpy(f_D, f_D2, size), 0);
    r = cuMemcpyDtoDAsync(f_D, f_D2, size, stream);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size);
    cuMemcpyDtoDAsync(f_D, f_D2, size, 0);
    // CHECK: r = (q_ct1.memcpy(f_D, f_D2, size), 0);
    r = cuMemcpyDtoDAsync(f_D, f_D2, size, 0);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size).wait();
    cuMemcpyDtoD(f_D, f_D2, size);
    // CHECK: r = (q_ct1.memcpy(f_D, f_D2, size).wait(), 0);
    r = cuMemcpyDtoD(f_D, f_D2, size);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size).wait();
    cuMemcpy(f_D, f_D2, size);
    // CHECK: CALL(q_ct1.memcpy(f_D, f_D2, size).wait());
    CALL(cuMemcpy(f_D, f_D2, size));
    // CHECK: r = (q_ct1.memcpy(f_D, f_D2, size).wait(), 0);
    r = cuMemcpy(f_D, f_D2, size);

    // CHECK: stream->memcpy(f_D, f_D2, size);
    cuMemcpyAsync(f_D, f_D2, size, stream);
    // CHECK: CALL(stream->memcpy(f_D, f_D2, size));
    CALL(cuMemcpyAsync(f_D, f_D2, size, stream));
    // CHECK: r = (stream->memcpy(f_D, f_D2, size), 0);
    r = cuMemcpyAsync(f_D, f_D2, size, stream);

    // CHECK: dpct::pitched_data cpy_from_data_ct1, cpy_to_data_ct1;
    // CHECK: sycl::id<3> cpy_from_pos_ct1(0, 0, 0), cpy_to_pos_ct1(0, 0, 0);
    // CHECK: sycl::range<3> cpy_size_ct1(1, 1, 1);
    CUDA_MEMCPY2D cpy;
    //
    cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
    // CHECK: cpy_to_data_ct1.set_data_ptr(f_A);
    cpy.dstHost = f_A;
    // CHECK: cpy_to_data_ct1.set_pitch(20);
    cpy.dstPitch = 20;
    // CHECK: cpy_to_pos_ct1[1] = 10;
    cpy.dstY = 10;
    // CHECK: cpy_to_pos_ct1[0] = 15;
    cpy.dstXInBytes = 15;

    //
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    // CHECK: cpy_from_data_ct1.set_data_ptr(f_D);
    cpy.srcDevice = f_D;
    // CHECK: cpy_from_data_ct1.set_pitch(20);
    cpy.srcPitch = 20;
    // CHECK: cpy_from_pos_ct1[1] = 10;
    cpy.srcY = 10;
    // CHECK: cpy_from_pos_ct1[0] = 15;
    cpy.srcXInBytes = 15;

    // CHECK: cpy_size_ct1[0] = 4;
    cpy.WidthInBytes = 4;
    // CHECK: cpy_size_ct1[1] = 7;
    cpy.Height = 7;

    // CHECK: dpct::dpct_memcpy(cpy_to_data_ct1, cpy_to_pos_ct1, cpy_from_data_ct1, cpy_from_pos_ct1, cpy_size_ct1);
    cuMemcpy2D(&cpy);
    // CHECK: dpct::async_dpct_memcpy(cpy_to_data_ct1, cpy_to_pos_ct1, cpy_from_data_ct1, cpy_from_pos_ct1, cpy_size_ct1, dpct::automatic, *stream);
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

    // CHECK: dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, advise);
    cuMemAdvise(devicePtr, count, advise, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError((dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, advise), 0));
    cuCheckError(cuMemAdvise(devicePtr, count, advise, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cu_err = (dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, advise), 0);
    cu_err = cuMemAdvise(devicePtr, count, advise, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, 0);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError((dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, 0), 0));
    cuCheckError(cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError((dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, 0), 0));
    cuCheckError(cuMemAdvise(devicePtr, count, (CUmem_advise)1, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError((dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, 0), 0));
    cuCheckError(cuMemAdvise(devicePtr, count, CUmem_advise(1), cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError((dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, 0), 0));
    cuCheckError(cuMemAdvise(devicePtr, count, static_cast<CUmem_advise>(1), cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cu_err = (dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, 0), 0);
    cu_err = cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: dpct::dev_mgr::instance().get_device(cudevice).default_queue().mem_advise(devicePtr, count, 0);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: dpct::cpu_device().default_queue().mem_advise(devicePtr, count, 0);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, CU_DEVICE_CPU);


    CUdeviceptr devPtr;
    CUresult curesult;
    // CHECK: stream->prefetch(devPtr, 100);
    cuMemPrefetchAsync (devPtr, 100, cudevice, stream);
    // CHECK: (*&stream)->prefetch(devPtr, 100);
    cuMemPrefetchAsync (devPtr, 100, cudevice, *&stream);
    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: curesult = (dpct::dev_mgr::instance().get_device(cudevice).default_queue().prefetch(devPtr, 100), 0);
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, NULL);
    // CHECK: dpct::dev_mgr::instance().get_device(cudevice).default_queue().prefetch(devPtr, 100);
    cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamPerThread);
    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: curesult = (dpct::dev_mgr::instance().get_device(cudevice).default_queue().prefetch(devPtr, 100), 0);
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamDefault);
    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: curesult = (dpct::dev_mgr::instance().get_device(cudevice).default_queue().prefetch(devPtr, 100), 0);
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamLegacy);
    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: curesult = (dpct::dev_mgr::instance().get_device(cudevice).default_queue().prefetch(devPtr, 100), 0);
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamPerThread);
    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError((dpct::dev_mgr::instance().get_device(cudevice).default_queue().prefetch(devPtr, 100), 0));
    cuCheckError(cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamDefault));
    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError((dpct::dev_mgr::instance().get_device(cudevice).default_queue().prefetch(devPtr, 100), 0));
    cuCheckError(cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamLegacy));
    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError((dpct::dev_mgr::instance().get_device(cudevice).default_queue().prefetch(devPtr, 100), 0));
    cuCheckError(cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamPerThread));

    // CHECK: dpct::pitched_data cpy2_from_data_ct1, cpy2_to_data_ct1;
    // CHECK: sycl::id<3> cpy2_from_pos_ct1(0, 0, 0), cpy2_to_pos_ct1(0, 0, 0);
    // CHECK: sycl::range<3> cpy2_size_ct1(1, 1, 1);
    CUDA_MEMCPY3D cpy2;

    CUarray ca;
    //
    cpy2.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    // CHECK: cpy2_to_data_ct1 = ca->to_pitched_data();
    cpy2.dstArray = ca;
    // CHECK: cpy2_to_data_ct1.set_pitch(5);
    cpy2.dstPitch = 5;
    // CHECK: cpy2_to_data_ct1.set_y(4);
    cpy2.dstHeight = 4;
    // CHECK: cpy2_to_pos_ct1[1] = 3;
    cpy2.dstY = 3;
    // CHECK: cpy2_to_pos_ct1[2] = 2;
    cpy2.dstZ = 2;
    // CHECK: cpy2_to_pos_ct1[0] = 1;
    cpy2.dstXInBytes = 1;
    //
    cpy2.dstLOD = 0;

    //
    cpy2.srcMemoryType = CU_MEMORYTYPE_HOST;
    // CHECK: cpy2_from_data_ct1.set_data_ptr(f_A);
    cpy2.srcHost = f_A;
    // CHECK: cpy2_from_data_ct1.set_pitch(5);
    cpy2.srcPitch = 5;
    // CHECK: cpy2_from_data_ct1.set_y(4);
    cpy2.srcHeight = 4;
    // CHECK: cpy2_from_pos_ct1[1] = 3;
    cpy2.srcY = 3;
    // CHECK: cpy2_from_pos_ct1[2] = 2;
    cpy2.srcZ = 2;
    // CHECK: cpy2_from_pos_ct1[0] = 1;
    cpy2.srcXInBytes = 1;
    //
    cpy2.srcLOD = 0;

    // CHECK: cpy2_size_ct1[0] = 3;
    cpy2.WidthInBytes = 3;
    // CHECK: cpy2_size_ct1[1] = 2;
    cpy2.Height = 2;
    // CHECK: cpy2_size_ct1[2] = 1;
    cpy2.Depth = 1;

    // CHECK: dpct::dpct_memcpy(cpy2_to_data_ct1, cpy2_to_pos_ct1, cpy2_from_data_ct1, cpy2_from_pos_ct1, cpy2_size_ct1);
    cuMemcpy3D(&cpy2);

    float *h_A = (float *)malloc(100);
    // CHECK:sycl::free(h_A, q_ct1);
    cuMemFreeHost(h_A);
    // CHECK:sycl::free(f_D, q_ct1);
    cuMemFree(f_D);

    unsigned int flags;
    int host;


    // CHECK: flags = 0;
    cuMemHostGetFlags(&flags, &host);
    // CHECK: cuCheckError((flags = 0, 0));
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
